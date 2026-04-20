# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import accelerate.utils.fsdp_utils as fsdp_utils
import builtins
import importlib
import inspect
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
_ORIGINAL_MINDSPEED_TE_CP_CLASS = None
_SUPPRESS_TRITON_TUNE_WARNING_ENV = 'SWIFT_SUPPRESS_TRITON_TUNE_WARNING'
_TRITON_TUNE_WARNING_PREFIX = '[WARNING] Please DO NOT tune args '
_QWEN3_NEXT_LINEAR_ATTENTION_PACKING_PATCHED = False


def _is_env_enabled(env_name: str) -> bool:
    return os.getenv(env_name, '0').lower() not in {'', '0', 'false', 'off'}


def _derive_cu_seqlens(position_ids: torch.Tensor | None) -> torch.Tensor | None:
    if position_ids is None or position_ids.ndim != 2 or position_ids.shape[0] != 1:
        return None
    flat_position_ids = position_ids[0]
    seq_starts = torch.nonzero(flat_position_ids == 0, as_tuple=False).flatten().to(dtype=torch.long)
    if seq_starts.numel() == 0:
        return None
    if seq_starts[0].item() != 0:
        seq_starts = torch.cat([seq_starts.new_zeros(1), seq_starts], dim=0)
    seq_end = seq_starts.new_tensor([flat_position_ids.shape[0]])
    return torch.cat([seq_starts, seq_end], dim=0)


def _function_accepts_argument(fn, arg_name: str) -> bool:
    try:
        return arg_name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def patch_qwen3_next_linear_attention_packing() -> None:
    global _QWEN3_NEXT_LINEAR_ATTENTION_PACKING_PATCHED
    if _QWEN3_NEXT_LINEAR_ATTENTION_PACKING_PATCHED:
        return

    try:
        model_mod = importlib.import_module('transformers.models.qwen3_next.modeling_qwen3_next')
    except ImportError as e:
        logger.warning(f'Failed to import Qwen3-Next modules for packing patching: {e}')
        return

    origin_decoder_forward = model_mod.Qwen3NextDecoderLayer.forward
    origin_gated_delta_forward = model_mod.Qwen3NextGatedDeltaNet.forward
    origin_torch_chunk_gated_delta_rule = model_mod.torch_chunk_gated_delta_rule

    def patched_torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=64,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
    ):
        if cu_seqlens is None:
            return origin_torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                chunk_size=chunk_size,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        if query.shape[0] != 1:
            raise ValueError(
                f'The batch size is expected to be 1 rather than {query.shape[0]} when using `cu_seqlens`.'
                'Please flatten variable-length inputs before processing.'
            )

        cu_seqlens = cu_seqlens.to(device=query.device, dtype=torch.long)
        num_sequences = cu_seqlens.numel() - 1
        if initial_state is not None and initial_state.shape[0] != num_sequences:
            raise ValueError(
                f'The number of initial states is expected to be equal to the number of input sequences, '
                f'i.e., {num_sequences} rather than {initial_state.shape[0]}.'
            )

        outputs = []
        final_states = []
        for seq_idx in range(num_sequences):
            start = int(cu_seqlens[seq_idx].item())
            end = int(cu_seqlens[seq_idx + 1].item())
            if end < start:
                raise ValueError(f'Invalid cu_seqlens: start={start}, end={end}.')
            seq_initial_state = None if initial_state is None else initial_state[seq_idx:seq_idx + 1]
            seq_output, seq_final_state = origin_torch_chunk_gated_delta_rule(
                query[:, start:end],
                key[:, start:end],
                value[:, start:end],
                g[:, start:end],
                beta[:, start:end],
                chunk_size=chunk_size,
                initial_state=seq_initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            outputs.append(seq_output)
            if output_final_state:
                final_states.append(seq_final_state)

        packed_output = torch.cat(outputs, dim=1) if outputs else value.new_empty(value.shape[0], 0, value.shape[-1])
        final_state = torch.cat(final_states, dim=0) if output_final_state and final_states else None
        return packed_output, final_state

    def patched_gated_delta_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ):
        if cu_seqlens is None:
            return origin_gated_delta_forward(
                self,
                hidden_states=hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
            )

        hidden_states = model_mod.apply_mask_to_padding_states(hidden_states, attention_mask)

        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx) and seq_len == 1

        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

        mixed_qkv = torch.cat((query, key, value), dim=-1)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        if use_precomputed_states:
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                conv_state = cache_params.update_conv_state(conv_state, self.layer_idx)
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            normalized_cu_seqlens = cu_seqlens.to(device=query.device, dtype=torch.long)
            chunk_kwargs = {
                'g': g,
                'beta': beta,
                'initial_state': None,
                'output_final_state': cache_params is not None,
                'use_qk_l2norm_in_kernel': True,
                'cu_seqlens': normalized_cu_seqlens,
            }
            if _function_accepts_argument(self.chunk_gated_delta_rule, 'cu_seqlens_cpu'):
                chunk_kwargs['cu_seqlens_cpu'] = normalized_cu_seqlens.cpu()
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(query, key, value, **chunk_kwargs)
        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)

        output = self.out_proj(core_attn_out)
        return output

    def patched_decoder_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        **kwargs,
    ):
        if self.layer_type != 'linear_attention':
            return origin_decoder_forward(
                self,
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        cu_seqlens = kwargs.get('cu_seq_lens_q')
        if cu_seqlens is None:
            cu_seqlens = _derive_cu_seqlens(position_ids)
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states
        return hidden_states

    model_mod.torch_chunk_gated_delta_rule = patched_torch_chunk_gated_delta_rule
    model_mod.Qwen3NextGatedDeltaNet.forward = patched_gated_delta_forward
    model_mod.Qwen3NextDecoderLayer.forward = patched_decoder_forward
    _QWEN3_NEXT_LINEAR_ATTENTION_PACKING_PATCHED = True
    logger.info('Patched Qwen3-Next linear attention to propagate packing `cu_seqlens` through native paths.')


def patch_qwen3_next_mindspeed_gated_delta_rule() -> None:
    patch_qwen3_next_linear_attention_packing()
    try:
        model_mod = importlib.import_module('transformers.models.qwen3_next.modeling_qwen3_next')
        mindspeed_chunk_mod = importlib.import_module('swift.model.chunk_gated_delta_rule')
    except ImportError as e:
        logger.warning(f'Failed to import Qwen3-Next MindSpeed gated delta modules for NPU patching: {e}')
        return
    model_mod.chunk_gated_delta_rule = mindspeed_chunk_mod.chunk_gated_delta_rule
    logger.info('Patched Qwen3-Next chunk_gated_delta_rule to swift.model.chunk_gated_delta_rule.chunk_gated_delta_rule.')


patch_qwen3_next_linear_attention_packing()
if _is_env_enabled('SWIFT_QWEN3_NEXT_USE_MINDSPEED_GATED_DELTA_RULE'):
    patch_qwen3_next_mindspeed_gated_delta_rule()


def suppress_triton_tune_warning() -> None:
    if os.getenv(_SUPPRESS_TRITON_TUNE_WARNING_ENV, '0').lower() in {'', '0', 'false', 'off'}:
        return

    try:
        jit_mod = importlib.import_module('triton.runtime.jit')
    except ImportError as e:
        logger.warning(f'Failed to import triton.runtime.jit for warning suppression: {e}')
        return

    origin_print = getattr(jit_mod, 'print', builtins.print)
    if getattr(origin_print, '__name__', '') == '_swift_suppressed_triton_print':
        return

    def _swift_suppressed_triton_print(*args, **kwargs):
        if args:
            first_arg = args[0]
            if isinstance(first_arg, str) and first_arg.startswith(_TRITON_TUNE_WARNING_PREFIX):
                return
        return origin_print(*args, **kwargs)

    jit_mod.print = _swift_suppressed_triton_print
    logger.info('Enabled Triton tune warning suppression for messages starting with `%s`.',
                _TRITON_TUNE_WARNING_PREFIX)


suppress_triton_tune_warning()


def _set_default_hccl_connect_timeout_for_npu() -> None:
    if 'HCCL_CONNECT_TIMEOUT' in os.environ:
        return

    os.environ['HCCL_CONNECT_TIMEOUT'] = _DEFAULT_NPU_HCCL_CONNECT_TIMEOUT
    logger.info(f'Set HCCL_CONNECT_TIMEOUT={_DEFAULT_NPU_HCCL_CONNECT_TIMEOUT} by default for NPU.')


_set_default_hccl_connect_timeout_for_npu()


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
