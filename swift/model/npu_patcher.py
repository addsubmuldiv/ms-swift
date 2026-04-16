# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import accelerate.utils.fsdp_utils as fsdp_utils
import importlib
import os
import torch
import torch.nn.functional as F
import torch_npu
from accelerate.accelerator import Accelerator
from functools import wraps
from torch import nn
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3_moe import modeling_qwen3_moe
from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe
from typing import Any

from swift.utils.logger import get_logger

logger = get_logger()

_DEFAULT_NPU_HCCL_CONNECT_TIMEOUT = '600'
_DEFAULT_QWEN_GATED_DELTA_FLA_CHUNK_SIZE = '32'
_QWEN_GATED_DELTA_FLA_KERNEL_ENV = 'SWIFT_QWEN3_NEXT_USE_NPU_FLA_KERNELS'
_FLA_GATED_DELTA_RULE_CHUNK_SIZE_ENV = 'FLA_GATED_DELTA_RULE_CHUNK_SIZE'
_ORIGINAL_MINDSPEED_TE_CP_CLASS = None
_QWEN_GATED_DELTA_FLA_PATCHED = False


def _set_default_hccl_connect_timeout_for_npu() -> None:
    if 'HCCL_CONNECT_TIMEOUT' in os.environ:
        return

    os.environ['HCCL_CONNECT_TIMEOUT'] = _DEFAULT_NPU_HCCL_CONNECT_TIMEOUT
    logger.info(f'Set HCCL_CONNECT_TIMEOUT={_DEFAULT_NPU_HCCL_CONNECT_TIMEOUT} by default for NPU.')


_set_default_hccl_connect_timeout_for_npu()


def _should_patch_qwen_gated_delta_fla_kernels() -> bool:
    return os.getenv(_QWEN_GATED_DELTA_FLA_KERNEL_ENV, '0') == '1'


def _set_default_qwen_gated_delta_fla_chunk_size_for_npu() -> None:
    if not _should_patch_qwen_gated_delta_fla_kernels() or _FLA_GATED_DELTA_RULE_CHUNK_SIZE_ENV in os.environ:
        return

    os.environ[_FLA_GATED_DELTA_RULE_CHUNK_SIZE_ENV] = _DEFAULT_QWEN_GATED_DELTA_FLA_CHUNK_SIZE
    logger.info(
        'Set %s=%s by default for Qwen gated-delta NPU FLA kernels.',
        _FLA_GATED_DELTA_RULE_CHUNK_SIZE_ENV,
        _DEFAULT_QWEN_GATED_DELTA_FLA_CHUNK_SIZE,
    )


def _resolve_qwen_gated_delta_fla_chunk_size() -> int:
    raw_chunk_size = os.getenv(_FLA_GATED_DELTA_RULE_CHUNK_SIZE_ENV, _DEFAULT_QWEN_GATED_DELTA_FLA_CHUNK_SIZE)
    try:
        chunk_size = int(raw_chunk_size)
    except ValueError as exc:
        raise ValueError(
            f'{_FLA_GATED_DELTA_RULE_CHUNK_SIZE_ENV} must be one of [16, 32, 64], got {raw_chunk_size!r}.') from exc
    if chunk_size not in {16, 32, 64}:
        raise ValueError(f'{_FLA_GATED_DELTA_RULE_CHUNK_SIZE_ENV} must be one of [16, 32, 64], got {chunk_size}.')
    return chunk_size


def patch_mindspeed_te_cp_implementation(megatron_args: dict[str, Any]) -> None:
    """
    Route NPU CP to the legacy MindSpeed TE adaptor when the new strategy factory
    only supports kvallgather.
    """
    try:
        import mindspeed.te.pytorch.attention.dot_product_attention.dot_product_attention as ms_te_dpa
        from mindspeed.core.context_parallel.adaptor import MindSpeedCPDotProductAttention
    except ImportError as e:
        logger.warning(f'Failed to import MindSpeed CP modules before repatch: {e}')
        return

    global _ORIGINAL_MINDSPEED_TE_CP_CLASS
    if _ORIGINAL_MINDSPEED_TE_CP_CLASS is None:
        _ORIGINAL_MINDSPEED_TE_CP_CLASS = getattr(ms_te_dpa, 'MindSpeedTEDotProductAttention', None)

    if _ORIGINAL_MINDSPEED_TE_CP_CLASS is None:
        logger.warning('MindSpeedTEDotProductAttention is unavailable before repatch; skip CP workaround.')
        return

    cp_algo = megatron_args.get('context_parallel_algo', 'megatron_cp_algo')
    use_legacy_cp_te = int(megatron_args.get('context_parallel_size', 1)) > 1 and cp_algo != 'kvallgather_cp_algo'
    target_cls = MindSpeedCPDotProductAttention if use_legacy_cp_te else _ORIGINAL_MINDSPEED_TE_CP_CLASS

    if getattr(ms_te_dpa, 'MindSpeedTEDotProductAttention', None) is target_cls:
        return

    ms_te_dpa.MindSpeedTEDotProductAttention = target_cls
    logger.info(
        'Patched MindSpeedTEDotProductAttention to %s for context_parallel_size=%s, context_parallel_algo=%s.',
        target_cls.__name__,
        megatron_args.get('context_parallel_size', 1),
        cp_algo,
    )


class NPUCastError(RuntimeError):
    """Raised when fp32 casting fails during NPU FSDP2 preparation."""


def _get_first_parameter(module: torch.nn.Module) -> torch.nn.Parameter | None:
    for param in module.parameters(recurse=True):
        return param
    return None


def _needs_fp32_cast_for_npu(
    module: torch.nn.Module,
    accelerator: Accelerator,
) -> bool:
    if accelerator.device.type != 'npu':
        return False

    param = _get_first_parameter(module)
    if param is None:
        return False

    return param.is_floating_point() and param.dtype != torch.float32


def _cast_to_fp32(module: torch.nn.Module) -> torch.nn.Module:
    """
    Cast module parameters to fp32.

    Assumes parameters are already on CPU or meta device.
    Only dtype is changed; device is preserved.
    """
    try:
        return module.to(torch.float32)
    except Exception as exc:
        raise NPUCastError(f'Failed to cast {module.__class__.__name__} to fp32.') from exc


# ----------------------------------------------------------------------
# Patch accelerate.utils.fsdp_utils.fsdp2_prepare_model
# ----------------------------------------------------------------------

_original_fsdp2_prepare_model = fsdp_utils.fsdp2_prepare_model


@wraps(_original_fsdp2_prepare_model)
def wrapped_fsdp2_prepare_model(
    accelerator: Accelerator,
    model: torch.nn.Module,
):
    if _needs_fp32_cast_for_npu(model, accelerator):
        model = _cast_to_fp32(model)

    return _original_fsdp2_prepare_model(accelerator, model)


fsdp_utils.fsdp2_prepare_model = wrapped_fsdp2_prepare_model

# ----------------------------------------------------------------------
# Patch Accelerator._prepare_fsdp2
# ----------------------------------------------------------------------

_original_prepare_fsdp2 = Accelerator._prepare_fsdp2


@wraps(_original_prepare_fsdp2)
def wrapped_prepare_fsdp2(
    self: Accelerator,
    *args,
    **kwargs,
):
    patched_args = [
        _cast_to_fp32(obj) if isinstance(obj, torch.nn.Module) and _needs_fp32_cast_for_npu(obj, self) else obj
        for obj in args
    ]

    return _original_prepare_fsdp2(self, *patched_args, **kwargs)


Accelerator._prepare_fsdp2 = wrapped_prepare_fsdp2


class NpuRMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]

    def extra_repr(self):
        return f'{tuple(self.weight.shape)}, eps={self.variance_epsilon}'


def npu_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed, k_embed


def npu_swiglu_forward(self, hidden_state):
    return self.down_proj(
        torch_npu.npu_swiglu(torch.cat((self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1), dim=-1))


class NpuGmmFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, group_list, split_size):
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list
        ctx.split_size = split_size

        outputs = torch_npu.npu_grouped_matmul([x], [weight], group_list=group_list, group_type=0, split_item=2)
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_outputs):
        x, weight = ctx.saved_tensors
        group_list = ctx.group_list
        wt = weight.permute(0, 2, 1)
        xt = x.permute(1, 0)
        dx = torch_npu.npu_grouped_matmul([grad_outputs], [wt], group_list=group_list, group_type=0, split_item=2)
        split_size = ctx.split_size
        xt_list = torch.split(xt, split_size, dim=1)
        grad_outputs_list = torch.split(grad_outputs, split_size, dim=0)
        with torch.npu.amp.autocast(enabled=False):
            dw = torch.stack([torch.matmul(xt_list[i], grad_outputs_list[i]) for i in range(len(xt_list))])

        return dx[0], dw, None, None


def npu_moe_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    # Loop over all available experts in the model and perform the computation on each expert
    # Concat all weights
    input_dtype = hidden_states.dtype
    up_weight_list = [e.up_proj.weight.t().to(input_dtype) for e in self.experts]
    gate_weight_list = [e.gate_proj.weight.t().to(input_dtype) for e in self.experts]
    down_weight_list = [e.down_proj.weight.t().to(input_dtype) for e in self.experts]
    w1 = torch.stack(up_weight_list)
    w2 = torch.stack(gate_weight_list)
    w3 = torch.stack(down_weight_list)

    # Copied from mindspeed moe_utils.py:permute
    routing_map = selected_experts
    flatten_indices = routing_map.view(-1)
    sorted_indices = torch.sort(flatten_indices.float(), stable=True)[1]
    permuted_tokens = hidden_states.index_select(0, sorted_indices // self.top_k)

    tokens_per_experts = torch.sum(expert_mask, dim=(1, 2))
    group_list = torch.cumsum(tokens_per_experts, dim=0)

    cpu_group_list = group_list.to('cpu', non_blocking=False)
    cpu_group_list = [0] + cpu_group_list.tolist()
    split_size = [cpu_group_list[i + 1] - cpu_group_list[i] for i in range(len(cpu_group_list) - 1)]

    up_res = NpuGmmFunction.apply(permuted_tokens, w1, group_list, split_size)
    gate_res = NpuGmmFunction.apply(permuted_tokens, w2, group_list, split_size)
    act_res = torch_npu.npu_swiglu(torch.cat([gate_res, up_res], dim=-1))
    down_res = NpuGmmFunction.apply(act_res, w3, group_list, split_size)

    probs = routing_weights
    num_unpermuted_tokens = probs.numel()
    topk = self.top_k
    permuted_tokens = down_res

    unpermuted_tokens = torch.zeros(
        [num_unpermuted_tokens, permuted_tokens.shape[-1]],
        dtype=permuted_tokens.dtype,
        device=permuted_tokens.device,
    )
    unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
    unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
    final_hidden_states = unpermuted_tokens.sum(dim=1).to(hidden_states.dtype)
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    return final_hidden_states, router_logits


class GmmFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, group_list):
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list

        fwd_output = torch_npu.npu_grouped_matmul([x], [weight],
                                                  bias=None,
                                                  group_list=group_list,
                                                  split_item=2,
                                                  group_type=0,
                                                  group_list_type=1)[0]
        return fwd_output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight = ctx.saved_tensors
        group_list = ctx.group_list

        weight = torch.transpose(weight, 1, 2)
        grad_input = torch_npu.npu_grouped_matmul([grad_output], [weight],
                                                  bias=None,
                                                  group_list=group_list,
                                                  split_item=2,
                                                  group_type=0,
                                                  group_list_type=1)[0]
        grad_weight = torch_npu.npu_grouped_matmul(
            [input_tensor.T],
            [grad_output],
            bias=None,
            group_list=group_list,
            split_item=3,
            group_type=2,
            group_list_type=1,
        )[0]
        return grad_input, grad_weight, None


class NpuMoeFused:

    @staticmethod
    def npu_moe_experts_forward(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor,
                                router_indices: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(hidden_states,
                                                                              router_indices.to(torch.int32))
        tokens_per_expert = torch.histc(router_indices, bins=self.num_experts, min=0, max=self.num_experts)
        intermediate_hidden_states = GmmFunction.apply(permuted_hidden_states, self.gate_up_proj, tokens_per_expert)
        intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)
        output = GmmFunction.apply(intermediate_activations, self.down_proj, tokens_per_expert)
        next_states = torch_npu.npu_moe_token_unpermute(output, row_ids_map, probs=routing_weights)
        next_states = next_states.view(batch_size, -1, self.hidden_size)
        return next_states

    @staticmethod
    def npu_moe_sparse_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)
        routed_out = self.experts(hidden_states, routing_weights, router_indices)
        return routed_out


def _setattr_path(root: Any, path: str, value: Any) -> None:
    current = root
    parts = path.split('.')
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], value)


def _apply_patch_map(root: Any, patch_map: dict[str, Any]) -> None:
    for path, value in patch_map.items():
        _setattr_path(root, path, value)


def patch_qwen_gated_delta_fla_implementation() -> None:
    global _QWEN_GATED_DELTA_FLA_PATCHED

    if _QWEN_GATED_DELTA_FLA_PATCHED or not _should_patch_qwen_gated_delta_fla_kernels():
        return

    _set_default_qwen_gated_delta_fla_chunk_size_for_npu()

    try:
        gated_delta_rule_pkg = importlib.import_module('fla.ops.gated_delta_rule')
        chunk_mod = importlib.import_module('fla.ops.gated_delta_rule.chunk')
        wy_fast_mod = importlib.import_module('fla.ops.gated_delta_rule.wy_fast')
        chunk_delta_h_mod = importlib.import_module('fla.ops.common.chunk_delta_h')
    except ImportError as e:
        logger.warning(f'Failed to import Qwen gated-delta FLA modules for NPU patching: {e}')
        return

    target_model_specs = (
        ('transformers.models.qwen3_next.modeling_qwen3_next', 'Qwen3-Next'),
        ('transformers.models.qwen3_5.modeling_qwen3_5', 'Qwen3.5'),
        ('transformers.models.qwen3_5_moe.modeling_qwen3_5_moe', 'Qwen3.5-MoE'),
    )
    target_model_modules = []
    for module_name, model_name in target_model_specs:
        try:
            target_model_modules.append((importlib.import_module(module_name), model_name))
        except ImportError as e:
            logger.debug('Skip gated-delta NPU patch for %s because %s is unavailable: %s', model_name, module_name, e)

    if not target_model_modules:
        logger.warning('No Qwen gated-delta model modules are available for NPU FLA patching.')
        return

    def patched_chunk_gated_delta_rule_bwd_dhu(
        q: torch.Tensor,
        k: torch.Tensor,
        w: torch.Tensor,
        do: torch.Tensor,
        dv: torch.Tensor,
        g: torch.Tensor | None = None,
        gk: torch.Tensor | None = None,
        h0: torch.Tensor | None = None,
        dht: torch.Tensor | None = None,
        scale: float | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_size: int = 64,
        chunk_indices: torch.LongTensor | None = None,
        use_exp2: bool = False,
        transpose_state_layout: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, H, K, V = *q.shape, do.shape[-1]
        BT = chunk_size
        assert K <= 256, 'current kernel does not support head dimension being larger than 256.'

        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = chunk_delta_h_mod.prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is None:
            N, NT, chunk_offsets = B, chunk_delta_h_mod.triton.cdiv(T, BT), None
        else:
            N = len(cu_seqlens) - 1
            NT = len(chunk_indices)
            chunk_offsets = chunk_delta_h_mod.prepare_chunk_offsets(cu_seqlens, BT)

        if transpose_state_layout:
            dh = q.new_empty(B, NT, H, V, K)
        else:
            dh = q.new_empty(B, NT, H, K, V)
        dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
        dv2 = torch.empty_like(dv)

        def grid(meta):
            return (chunk_delta_h_mod.triton.cdiv(V, meta['BV']), N * H)

        chunk_delta_h_mod.chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64[grid](
            q=q,
            k=k,
            w=w,
            g=g,
            gk=gk,
            dht=dht,
            dh0=dh0,
            do=do,
            dh=dh,
            dv=dv,
            dv2=dv2,
            cu_seqlens=cu_seqlens,
            chunk_offsets=chunk_offsets,
            scale=scale,
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            USE_EXP2=use_exp2,
            TRANSPOSE_STATE=transpose_state_layout,
        )
        return dh, dh0, dv2

    def patched_prepare_wy_repr_bwd(
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        dw: torch.Tensor,
        du: torch.Tensor,
        g: torch.Tensor = None,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, H, K, V = *k.shape, v.shape[-1]
        BT = A.shape[-1]
        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = wy_fast_mod.prepare_chunk_indices(cu_seqlens, BT)
        NT = wy_fast_mod.triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
        CONST_TILING = 64 if wy_fast_mod.check_shared_mem() else 32
        BK = min(max(wy_fast_mod.triton.next_power_of_2(K), 16), CONST_TILING)
        BV = min(max(wy_fast_mod.triton.next_power_of_2(V), 16), CONST_TILING)

        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dg = torch.empty_like(g) if g is not None else None
        db = torch.empty_like(beta)
        wy_fast_mod.prepare_wy_repr_bwd_kernel[(NT, B * H)](
            k=k,
            v=v,
            beta=beta,
            g=g,
            A=A,
            dw=dw,
            du=du,
            dk=dk,
            dv=dv,
            db=db,
            dg=dg,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
        )
        return dk, dv, db, dg

    def patched_chunk_gated_delta_rule_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        cp_context=None,
        chunk_indices: torch.LongTensor | None = None,
        transpose_state_layout: bool = False,
        chunk_size: int = 64,
    ):
        g = chunk_mod.chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
        A = chunk_mod.chunk_scaled_dot_kkt_fwd(
            k=k,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            output_dtype=torch.float32,
            chunk_indices=chunk_indices,
        )
        A = chunk_mod.solve_tril(
            A=A,
            cu_seqlens=cu_seqlens,
            output_dtype=k.dtype,
            chunk_indices=chunk_indices,
        )
        w, u = chunk_mod.recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=A,
            g=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

        if cp_context is not None:
            initial_state = chunk_mod.chunk_gated_delta_rule_fwd_h_pre_process(
                k=k,
                w=w,
                u=u,
                g=g,
                chunk_size=chunk_size,
                cu_seqlens=cu_seqlens,
                initial_state=initial_state,
                context=cp_context,
                transpose_state_layout=transpose_state_layout,
            )

        h, v_new, final_state = chunk_mod.chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            transpose_state_layout=transpose_state_layout,
        )

        if cp_context is not None:
            initial_state = chunk_mod.compress_h0(initial_state, context=cp_context)

        o = chunk_mod.chunk_fwd_o(
            q=q,
            k=k,
            v=v_new,
            h=h,
            g=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            transpose_state_layout=transpose_state_layout,
        )
        return g, o, A, final_state, initial_state

    def patched_chunk_gated_delta_rule_bwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        do: torch.Tensor,
        dht: torch.Tensor,
        cu_seqlens: torch.LongTensor | None = None,
        cp_context=None,
        chunk_indices: torch.LongTensor | None = None,
        transpose_state_layout: bool = False,
    ):
        chunk_size = A.shape[-1]
        w, u = chunk_mod.recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=A,
            g=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

        if cp_context is not None:
            initial_state = chunk_mod.expand_h0(initial_state, context=cp_context)

        h, v_new, _ = chunk_mod.chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            transpose_state_layout=transpose_state_layout,
        )
        dv = chunk_mod.chunk_bwd_dv_local(
            q=q,
            k=k,
            g=g,
            do=do,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )

        if cp_context is not None:
            dht, initial_state = chunk_mod.chunk_gated_delta_rule_bwd_dhu_pre_process(
                q=q,
                k=k,
                w=w,
                do=do,
                dv=dv,
                g=g,
                scale=scale,
                chunk_size=chunk_size,
                cu_seqlens=cu_seqlens,
                dht=dht,
                initial_state=initial_state,
                context=cp_context,
                transpose_state_layout=transpose_state_layout,
            )

        dh, dh0, dv = patched_chunk_gated_delta_rule_bwd_dhu(
            q=q,
            k=k,
            w=w,
            g=g,
            h0=initial_state,
            dht=dht,
            do=do,
            dv=dv,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            transpose_state_layout=transpose_state_layout,
        )
        dq, dk, dw, dg = chunk_mod.chunk_bwd_dqkwg(
            q=q,
            k=k,
            v=v_new,
            w=w,
            g=g,
            h=h,
            dv=dv,
            do=do,
            dh=dh,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            transpose_state_layout=transpose_state_layout,
        )
        dk2, dv, db, dg2 = patched_prepare_wy_repr_bwd(
            k=k,
            v=v,
            beta=beta,
            g=g,
            A=A,
            dw=dw,
            du=dv,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        dk.add_(dk2)
        dg.add_(dg2)
        dg = chunk_mod.chunk_local_cumsum(
            dg,
            chunk_size=chunk_size,
            reverse=True,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        return dq, dk, dv, db, dg, dh0

    class PatchedChunkGatedDeltaRuleFunction(torch.autograd.Function):

        @staticmethod
        @chunk_mod.input_guard
        @chunk_mod.autocast_custom_fwd
        def forward(
            ctx,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
            scale: float,
            initial_state: torch.Tensor,
            output_final_state: bool,
            cu_seqlens: torch.LongTensor | None = None,
            cu_seqlens_cpu: torch.LongTensor | None = None,
            use_qk_l2norm_in_kernel: bool = False,
            cp_context=None,
            transpose_state_layout: bool = False,
        ):
            chunk_size = _resolve_qwen_gated_delta_fla_chunk_size()
            q_rstd, k_rstd = None, None
            if use_qk_l2norm_in_kernel:
                q, q_rstd = chunk_mod.l2norm_fwd(q)
                k, k_rstd = chunk_mod.l2norm_fwd(k)

            chunk_indices = chunk_mod.prepare_chunk_indices(
                cu_seqlens,
                chunk_size,
                cu_seqlens_cpu=cu_seqlens_cpu,
            ) if cu_seqlens is not None else None
            g, o, A, final_state, initial_state = patched_chunk_gated_delta_rule_fwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                cp_context=cp_context,
                chunk_indices=chunk_indices,
                transpose_state_layout=transpose_state_layout,
                chunk_size=chunk_size,
            )
            ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens, chunk_indices)
            ctx.scale = scale
            ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
            ctx.cp_context = cp_context
            ctx.transpose_state_layout = transpose_state_layout
            return o.to(q.dtype), final_state

        @staticmethod
        @chunk_mod.input_guard
        @chunk_mod.autocast_custom_bwd
        def backward(
            ctx,
            do: torch.Tensor,
            dht: torch.Tensor,
        ):
            q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens, chunk_indices = ctx.saved_tensors
            dq, dk, dv, db, dg, dh0 = patched_chunk_gated_delta_rule_bwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A=A,
                scale=ctx.scale,
                initial_state=initial_state,
                do=do,
                dht=dht,
                cu_seqlens=cu_seqlens,
                cp_context=ctx.cp_context,
                chunk_indices=chunk_indices,
                transpose_state_layout=ctx.transpose_state_layout,
            )
            if ctx.use_qk_l2norm_in_kernel:
                dq = chunk_mod.l2norm_bwd(q, q_rstd, dq)
                dk = chunk_mod.l2norm_bwd(k, k_rstd, dk)
            return dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta), None, dh0, None, None, None, None, None, None

    @torch.compiler.disable
    def patched_chunk_gated_delta_rule(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float = None,
        initial_state: torch.Tensor = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        cp_context=None,
        transpose_state_layout: bool = False,
        **kwargs,
    ):
        if 'head_first' in kwargs:
            chunk_mod.warnings.warn(
                'head_first is deprecated and will be removed in a future version. '
                'Please use head_first=False for now instead.',
            )

        if cp_context is not None:
            assert initial_state is None, 'Initial state is not supported for CP'
            assert output_final_state is False, 'Output final state is not supported for CP'
            assert cp_context.cu_seqlens is not None, 'cu_seqlens is required for CP'
            cu_seqlens = cp_context.cu_seqlens
            if cp_context.cu_seqlens_cpu is not None:
                cu_seqlens_cpu = cp_context.cu_seqlens_cpu

        if cu_seqlens is not None:
            if q.shape[0] != 1:
                raise ValueError(
                    f'The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`.'
                    f'Please flatten variable-length inputs before processing.',
                )
            if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
                raise ValueError(
                    'The number of initial states is expected to be equal to the number of input sequences, '
                    f'i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.',
                )
        if scale is None:
            scale = k.shape[-1] ** -0.5
        o, final_state = PatchedChunkGatedDeltaRuleFunction.apply(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            output_final_state,
            cu_seqlens,
            cu_seqlens_cpu,
            use_qk_l2norm_in_kernel,
            cp_context,
            transpose_state_layout,
        )
        return o, final_state

    chunk_delta_h_mod.chunk_gated_delta_rule_bwd_dhu = patched_chunk_gated_delta_rule_bwd_dhu
    wy_fast_mod.prepare_wy_repr_bwd = patched_prepare_wy_repr_bwd
    chunk_mod.prepare_wy_repr_bwd = patched_prepare_wy_repr_bwd
    chunk_mod.chunk_gated_delta_rule_fwd = patched_chunk_gated_delta_rule_fwd
    chunk_mod.chunk_gated_delta_rule_bwd = patched_chunk_gated_delta_rule_bwd
    chunk_mod.ChunkGatedDeltaRuleFunction = PatchedChunkGatedDeltaRuleFunction
    chunk_mod.chunk_gated_delta_rule = patched_chunk_gated_delta_rule
    gated_delta_rule_pkg.chunk_gated_delta_rule = patched_chunk_gated_delta_rule
    patched_model_names = []
    for model_mod, model_name in target_model_modules:
        model_mod.chunk_gated_delta_rule = patched_chunk_gated_delta_rule
        patched_model_names.append(model_name)

    _QWEN_GATED_DELTA_FLA_PATCHED = True
    logger.info(
        'Patched Qwen gated-delta FLA chunk_size=%s for %s in npu_patcher.',
        _resolve_qwen_gated_delta_fla_chunk_size(),
        ', '.join(patched_model_names),
    )


_PATCH_TABLE: tuple[tuple[Any, dict[str, Any]], ...] = (
    (
        modeling_qwen2,
        {
            'Qwen2RMSNorm': NpuRMSNorm,
            'apply_rotary_pos_emb': npu_apply_rotary_pos_emb,
            'Qwen2MLP.forward': npu_swiglu_forward,
        },
    ),
    (
        modeling_qwen3,
        {
            'Qwen3RMSNorm': NpuRMSNorm,
            'apply_rotary_pos_emb': npu_apply_rotary_pos_emb,
            'Qwen3MLP.forward': npu_swiglu_forward,
        },
    ),
    (
        modeling_qwen3_moe,
        {
            'Qwen3MoeRMSNorm': NpuRMSNorm,
            'apply_rotary_pos_emb': npu_apply_rotary_pos_emb,
            'Qwen3MoeSparseMoeBlock.forward': npu_moe_block_forward,
        },
    ),
    (
        modeling_qwen3_vl_moe,
        {
            'Qwen3VLMoeTextExperts.forward': NpuMoeFused.npu_moe_experts_forward,
            'Qwen3VLMoeTextSparseMoeBlock.forward': NpuMoeFused.npu_moe_sparse_block_forward,
            'Qwen3VLMoeTextRMSNorm': NpuRMSNorm,
            'apply_rotary_pos_emb': npu_apply_rotary_pos_emb,
        },
    ),
)

for _module, _patch_map in _PATCH_TABLE:
    _apply_patch_map(_module, _patch_map)

patch_qwen_gated_delta_fla_implementation()