# Some code borrowed from the awesome work: https://github.com/zhuzilin/ring-flash-attention
# Copyright (c) ModelScope Contributors. All rights reserved.
from functools import cache

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .utils import RingComm

_NPU_BLOCK_MASK_SIZE = 2048
_NPU_FULL_TOKENS = 2147483647


def is_npu_tensor(tensor: torch.Tensor) -> bool:
    return tensor.device.type == 'npu'


def _is_npu_tensor(tensor: torch.Tensor) -> bool:
    return is_npu_tensor(tensor)


def cu_seqlens_to_actual_seq(cu_seqlens: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(x) for x in cu_seqlens[1:].detach().cpu().tolist())


def _cu_seqlens_to_actual_seq(cu_seqlens: torch.Tensor) -> tuple[int, ...]:
    return cu_seqlens_to_actual_seq(cu_seqlens)


@cache
def _get_npu_causal_mask_cpu() -> torch.Tensor:
    return torch.triu(torch.ones((_NPU_BLOCK_MASK_SIZE, _NPU_BLOCK_MASK_SIZE), dtype=torch.bool), diagonal=1)


def _get_npu_causal_mask(device: torch.device) -> torch.Tensor:
    return _get_npu_causal_mask_cpu().to(device=device)


def _normalize_window_size(window_size):
    if window_size is None:
        return -1, -1
    return window_size


def _get_npu_sparse_params(causal: bool, window_size, device: torch.device) -> dict:
    window_size = _normalize_window_size(window_size)
    if window_size != (-1, -1):
        left, right = window_size
        left = _NPU_FULL_TOKENS if left < 0 else int(left)
        right = _NPU_FULL_TOKENS if right < 0 else int(right)
        if causal:
            right = 0
        return {
            'atten_mask': _get_npu_causal_mask(device),
            'sparse_mode': 4,
            'pre_tockens': left,
            'next_tockens': right,
        }
    if causal:
        return {
            'atten_mask': _get_npu_causal_mask(device),
            'sparse_mode': 3,
            'pre_tockens': _NPU_FULL_TOKENS,
            'next_tockens': _NPU_FULL_TOKENS,
        }
    return {
        'atten_mask': None,
        'sparse_mode': 0,
        'pre_tockens': _NPU_FULL_TOKENS,
        'next_tockens': _NPU_FULL_TOKENS,
    }


def reshape_npu_lse(lse: torch.Tensor, seqlen_q: int, num_heads: int) -> torch.Tensor:
    if lse.dim() == 2:
        if lse.shape == (num_heads, seqlen_q):
            return lse.contiguous()
        if lse.shape == (seqlen_q, num_heads):
            return lse.transpose(0, 1).contiguous()
    elif lse.dim() == 3:
        # Ascend ring-attention related outputs commonly use an extra size-8
        # trailing axis whose values are duplicated for numerical updates.
        if lse.shape[-1] == 8:
            lse = lse[..., 0]
            if lse.shape == (seqlen_q, num_heads):
                return lse.transpose(0, 1).contiguous()
            if lse.shape == (num_heads, seqlen_q):
                return lse.contiguous()
        if lse.shape[0] == seqlen_q:
            return lse.permute(1, 2, 0).reshape(num_heads, seqlen_q).contiguous()
        if lse.shape[1] == seqlen_q:
            return lse.permute(0, 2, 1).reshape(num_heads, seqlen_q).contiguous()
    raise RuntimeError(f'Unexpected NPU lse shape {tuple(lse.shape)} for seqlen_q={seqlen_q}, num_heads={num_heads}')


def _reshape_npu_lse(lse: torch.Tensor, seqlen_q: int, num_heads: int) -> torch.Tensor:
    return reshape_npu_lse(lse, seqlen_q, num_heads)


def _get_npu_attention_common_kwargs(
    q: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    softmax_scale: float,
    dropout_p: float,
    causal: bool,
    window_size,
    deterministic: bool,
) -> dict:
    sparse_params = _get_npu_sparse_params(causal, window_size, q.device)
    return {
        'head_num': q.shape[1],
        'input_layout': 'TND',
        'scale_value': softmax_scale or q.shape[-1]**(-0.5),
        'keep_prob': 1. - dropout_p,
        'actual_seq_qlen': cu_seqlens_to_actual_seq(cu_seqlens_q),
        'actual_seq_kvlen': cu_seqlens_to_actual_seq(cu_seqlens_kv),
        'sync': bool(deterministic and dropout_p > 0),
        **sparse_params,
    }


def _call_npu_fusion_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    softmax_scale: float,
    dropout_p: float,
    causal: bool,
    window_size,
    deterministic: bool,
    return_ctx: bool = False,
):
    import torch_npu

    common_kwargs = _get_npu_attention_common_kwargs(
        q,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
    )
    params = {
        'query': q,
        'key': k,
        'value': v,
        'scale': common_kwargs['scale_value'],
        'softmax_layout': 'TND',
        **{k: v for k, v in common_kwargs.items() if k != 'scale_value'},
    }
    used_softmax_layout = True
    try:
        outputs = torch_npu.npu_fusion_attention(**params)
    except TypeError as exc:
        if 'softmax_layout' not in str(exc):
            raise
        params.pop('softmax_layout', None)
        used_softmax_layout = False
        outputs = torch_npu.npu_fusion_attention(**params)
    if not return_ctx:
        return outputs

    block_out, softmax_max, softmax_sum, attention_in, seed, offset, numels = outputs
    ctx = {
        **common_kwargs,
        'softmax_max': softmax_max,
        'softmax_sum': softmax_sum,
        'attention_in': attention_in,
        'seed': seed,
        'offset': offset,
        'numels': numels,
        'softmax_layout': 'TND' if used_softmax_layout else '',
    }
    return outputs, ctx


def _call_npu_fusion_attention_grad(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ctx: dict,
):
    import torch_npu

    if not hasattr(torch_npu, 'npu_fusion_attention_grad'):
        raise AttributeError('torch_npu.npu_fusion_attention_grad is not available')

    params = {
        'query': q,
        'key': k,
        'value': v,
        'dy': dout,
        'head_num': ctx['head_num'],
        'input_layout': ctx['input_layout'],
        'atten_mask': ctx['atten_mask'],
        'softmax_max': ctx['softmax_max'],
        'softmax_sum': ctx['softmax_sum'],
        'attention_in': ctx['attention_in'] if torch.is_tensor(ctx['attention_in']) and ctx['attention_in'].numel() > 0 else None,
        'scale_value': ctx['scale_value'],
        'keep_prob': ctx['keep_prob'],
        'pre_tockens': ctx['pre_tockens'],
        'next_tockens': ctx['next_tockens'],
        'seed': ctx['seed'],
        'offset': ctx['offset'],
        'numels': ctx['numels'],
        'actual_seq_qlen': ctx['actual_seq_qlen'],
        'actual_seq_kvlen': ctx['actual_seq_kvlen'],
        'sparse_mode': ctx['sparse_mode'],
        'sync': ctx['sync'],
    }
    if ctx.get('softmax_layout'):
        params['softmax_layout'] = ctx['softmax_layout']
    try:
        return torch_npu.npu_fusion_attention_grad(**params)
    except TypeError as exc:
        if 'softmax_layout' not in str(exc):
            raise
        params.pop('softmax_layout', None)
        return torch_npu.npu_fusion_attention_grad(**params)


def _get_npu_manual_attention_mask(causal: bool, window_size, q_len: int, k_len: int, device: torch.device) -> torch.Tensor | None:
    window_size = _normalize_window_size(window_size)
    offset = k_len - q_len
    q_idx = torch.arange(q_len, device=device).unsqueeze(1)
    k_idx = torch.arange(k_len, device=device).unsqueeze(0)
    mask = None
    if causal:
        mask = k_idx > (q_idx + offset)
    if window_size != (-1, -1):
        left, right = window_size
        if left >= 0:
            left_mask = k_idx < (q_idx + offset - left)
            mask = left_mask if mask is None else (mask | left_mask)
        if right >= 0:
            right_mask = k_idx > (q_idx + offset + right)
            mask = right_mask if mask is None else (mask | right_mask)
    return mask


def _normalize_softmax_lse(softmax_lse: torch.Tensor, q_tokens: int, num_heads: int) -> torch.Tensor:
    if softmax_lse.dim() == 3:
        softmax_lse = softmax_lse.squeeze(0)
    if softmax_lse.dim() != 2:
        raise RuntimeError(f'Unexpected softmax_lse shape: {tuple(softmax_lse.shape)}')
    if softmax_lse.shape == (num_heads, q_tokens):
        return softmax_lse.transpose(0, 1).contiguous()
    if softmax_lse.shape == (q_tokens, num_heads):
        return softmax_lse.contiguous()
    raise RuntimeError(
        f'Unexpected softmax_lse shape: {tuple(softmax_lse.shape)} for q_tokens={q_tokens}, num_heads={num_heads}')


def manual_varlen_attention_backward(dout,
                                     q,
                                     k,
                                     v,
                                     out,
                                     softmax_lse,
                                     cu_seqlens_q,
                                     cu_seqlens_kv,
                                     softmax_scale,
                                     causal,
                                     window_size):
    scale = softmax_scale or q.shape[-1]**(-0.5)
    num_heads_q = q.shape[1]
    num_heads_kv = k.shape[1]
    groups = num_heads_q // num_heads_kv
    assert groups * num_heads_kv == num_heads_q
    softmax_lse = _normalize_softmax_lse(softmax_lse, q.shape[0], num_heads_q)

    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)

    for i in range(len(cu_seqlens_q) - 1):
        q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        k_start, k_end = cu_seqlens_kv[i].item(), cu_seqlens_kv[i + 1].item()
        q_seq = q[q_start:q_end].to(torch.float32)
        k_seq = k[k_start:k_end].to(torch.float32)
        v_seq = v[k_start:k_end].to(torch.float32)
        out_seq = out[q_start:q_end].to(torch.float32)
        dout_seq = dout[q_start:q_end].to(torch.float32)
        lse_seq = softmax_lse[q_start:q_end].transpose(0, 1).unsqueeze(-1).to(torch.float32)

        if groups > 1:
            k_seq_expanded = k_seq.repeat_interleave(groups, dim=1)
            v_seq_expanded = v_seq.repeat_interleave(groups, dim=1)
        else:
            k_seq_expanded = k_seq
            v_seq_expanded = v_seq

        scores = torch.einsum('qhd,khd->hqk', q_seq, k_seq_expanded) * scale
        mask = _get_npu_manual_attention_mask(causal, window_size, q_seq.shape[0], k_seq.shape[0], scores.device)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0), torch.finfo(scores.dtype).min)
        probs = torch.exp(scores - lse_seq)
        delta = (out_seq * dout_seq).sum(dim=-1).transpose(0, 1).unsqueeze(-1)

        d_v_expanded = torch.einsum('hqk,qhd->khd', probs, dout_seq)
        d_probs = torch.einsum('qhd,khd->hqk', dout_seq, v_seq_expanded)
        d_scores = probs * (d_probs - delta)
        d_q = torch.einsum('hqk,khd->qhd', d_scores, k_seq_expanded) * scale
        d_k_expanded = torch.einsum('hqk,qhd->khd', d_scores, q_seq) * scale

        if groups > 1:
            d_k = d_k_expanded.reshape(k_seq.shape[0], num_heads_kv, groups, k_seq.shape[-1]).sum(dim=2)
            d_v = d_v_expanded.reshape(v_seq.shape[0], num_heads_kv, groups, v_seq.shape[-1]).sum(dim=2)
        else:
            d_k = d_k_expanded
            d_v = d_v_expanded

        dq[q_start:q_end] = d_q
        dk[k_start:k_end] = d_k
        dv[k_start:k_end] = d_v
    return dq, dk, dv


def _manual_varlen_attention_backward(dout,
                                      q,
                                      k,
                                      v,
                                      out,
                                      softmax_lse,
                                      cu_seqlens_q,
                                      cu_seqlens_kv,
                                      softmax_scale,
                                      causal,
                                      window_size):
    return manual_varlen_attention_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_kv,
        softmax_scale,
        causal,
        window_size,
    )


def manual_varlen_lse_backward(dlse, q, k, cu_seqlens_q, cu_seqlens_kv, softmax_scale, causal, window_size):
    scale = softmax_scale or q.shape[-1]**(-0.5)
    num_heads_q = q.shape[1]
    num_heads_kv = k.shape[1]
    groups = num_heads_q // num_heads_kv
    assert groups * num_heads_kv == num_heads_q

    if dlse.dim() == 3:
        dlse = dlse.squeeze(-1)
    assert dlse.shape == (q.shape[0], num_heads_q)

    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)

    for i in range(len(cu_seqlens_q) - 1):
        q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        k_start, k_end = cu_seqlens_kv[i].item(), cu_seqlens_kv[i + 1].item()
        q_seq = q[q_start:q_end].to(torch.float32)
        k_seq = k[k_start:k_end].to(torch.float32)
        dlse_seq = dlse[q_start:q_end].transpose(0, 1).to(torch.float32)

        if groups > 1:
            k_seq_expanded = k_seq.repeat_interleave(groups, dim=1)
        else:
            k_seq_expanded = k_seq

        scores = torch.einsum('qhd,khd->hqk', q_seq, k_seq_expanded) * scale
        mask = _get_npu_manual_attention_mask(causal, window_size, q_seq.shape[0], k_seq.shape[0], scores.device)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0), torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        d_scores = dlse_seq.unsqueeze(-1) * probs
        d_q = torch.einsum('hqk,khd->qhd', d_scores, k_seq_expanded) * scale
        d_k_expanded = torch.einsum('hqk,qhd->khd', d_scores, q_seq) * scale

        if groups > 1:
            d_k = d_k_expanded.reshape(k_seq.shape[0], num_heads_kv, groups, k_seq.shape[-1]).sum(dim=2)
        else:
            d_k = d_k_expanded

        dq[q_start:q_end] += d_q
        dk[k_start:k_end] += d_k
    return dq, dk


def _manual_varlen_lse_backward(dlse, q, k, cu_seqlens_q, cu_seqlens_kv, softmax_scale, causal, window_size):
    return manual_varlen_lse_backward(dlse, q, k, cu_seqlens_q, cu_seqlens_kv, softmax_scale, causal, window_size)


def _manual_varlen_attention_forward(q, k, v, cu_seqlens_q, cu_seqlens_kv, softmax_scale, causal, window_size):
    scale = softmax_scale or q.shape[-1]**(-0.5)
    num_heads_q = q.shape[1]
    num_heads_kv = k.shape[1]
    groups = num_heads_q // num_heads_kv
    assert groups * num_heads_kv == num_heads_q

    outputs = []
    lses = []
    for i in range(len(cu_seqlens_q) - 1):
        q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        k_start, k_end = cu_seqlens_kv[i].item(), cu_seqlens_kv[i + 1].item()
        q_seq = q[q_start:q_end].to(torch.float32)
        k_seq = k[k_start:k_end].to(torch.float32)
        v_seq = v[k_start:k_end].to(torch.float32)

        if groups > 1:
            k_seq_expanded = k_seq.repeat_interleave(groups, dim=1)
            v_seq_expanded = v_seq.repeat_interleave(groups, dim=1)
        else:
            k_seq_expanded = k_seq
            v_seq_expanded = v_seq

        scores = torch.einsum('qhd,khd->hqk', q_seq, k_seq_expanded) * scale
        mask = _get_npu_manual_attention_mask(causal, window_size, q_seq.shape[0], k_seq.shape[0], scores.device)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0), torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum('hqk,khd->qhd', probs, v_seq_expanded))
        lses.append(torch.logsumexp(scores, dim=-1))

    return torch.cat(outputs, dim=0).contiguous(), torch.cat(lses, dim=1).contiguous()


def _all_gather_step_grads(step_grads: torch.Tensor, process_group) -> list[torch.Tensor]:
    gathered = [torch.empty_like(step_grads) for _ in range(dist.get_world_size(process_group))]
    dist.all_gather(gathered, step_grads.contiguous(), group=process_group)
    return gathered


def _squeeze_batch(*tensors):
    squeezed = []
    for tensor in tensors:
        if tensor.shape[0] == 1:
            squeezed.append(tensor.squeeze(0))
        else:
            squeezed.append(tensor)
    return tuple(squeezed)


def _update_out_and_lse(out, lse, block_out, block_lse):
    if out is None:
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
        sig_diff = None
    else:
        block_out = block_out.to(torch.float32)
        block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

        diff = block_lse - lse
        sig_diff = torch.sigmoid(diff)

        out = out - sig_diff * (out - block_out)
        lse = lse - F.logsigmoid(lse - block_lse)
    return out, lse, sig_diff


def zigzag_ring_flash_attn_varlen_backward_exact(
    process_group,
    dout,
    q,
    k,
    v,
    cu_seqlens,
    max_seqlen,
    half_index0,
    half_index1,
    softmax_scale,
    window_size,
):
    kv_comm = RingComm(process_group)
    dout, q, k, v = _squeeze_batch(dout, q, k, v)
    cu_seqlens = cu_seqlens // kv_comm.world_size
    max_seqlen = max_seqlen // kv_comm.world_size
    block_seq_len = q.shape[0] // 2
    half_cu_seqlens = cu_seqlens // 2

    def _get_block_cu_seqlens(seqlen_q, seqlen_kv):
        cu_seqlens_q = half_cu_seqlens if seqlen_q == block_seq_len else cu_seqlens
        cu_seqlens_kv = half_cu_seqlens if seqlen_kv == block_seq_len else cu_seqlens
        return cu_seqlens_q, cu_seqlens_kv

    with torch.enable_grad():
        q_replay = q.detach().requires_grad_(True)
        current_k = k.detach().requires_grad_(True)
        current_v = v.detach().requires_grad_(True)
        step_ks = []
        step_vs = []
        merged_out = None
        merged_lse = None

        for step in range(kv_comm.world_size):
            step_ks.append(current_k)
            step_vs.append(current_v)
            if step + 1 != kv_comm.world_size:
                next_k, next_v = kv_comm.send_recv_kv(current_k.detach(), current_v.detach())

            if step == 0:
                block_q = q_replay
                block_k = current_k
                block_v = current_v
                block_causal = True
            elif step <= kv_comm.rank:
                block_q = q_replay
                block_k = current_k[half_index0]
                block_v = current_v[half_index0]
                block_causal = False
            else:
                block_q = q_replay[half_index1]
                block_k = current_k
                block_v = current_v
                block_causal = False

            cu_seqlens_q, cu_seqlens_kv = _get_block_cu_seqlens(block_q.shape[0], block_k.shape[0])
            block_out, block_lse = _manual_varlen_attention_forward(
                block_q,
                block_k,
                block_v,
                cu_seqlens_q,
                cu_seqlens_kv,
                softmax_scale,
                block_causal,
                window_size,
            )

            if step == 0 or step <= kv_comm.rank:
                merged_out, merged_lse, _ = _update_out_and_lse(merged_out, merged_lse, block_out, block_lse)
            else:
                merged_out[half_index1], merged_lse[half_index1], _ = _update_out_and_lse(
                    merged_out[half_index1],
                    merged_lse[half_index1],
                    block_out,
                    block_lse,
                )

            if step + 1 != kv_comm.world_size:
                kv_comm.wait()
                current_k = next_k.detach().requires_grad_(True)
                current_v = next_v.detach().requires_grad_(True)

        grads = torch.autograd.grad(
            merged_out,
            [q_replay, *step_ks, *step_vs],
            grad_outputs=dout.to(merged_out.dtype),
        )

    dq = grads[0].to(torch.float32)
    num_steps = kv_comm.world_size
    step_dk = torch.stack([grad.to(torch.float32) for grad in grads[1:1 + num_steps]], dim=0)
    step_dv = torch.stack([grad.to(torch.float32) for grad in grads[1 + num_steps:]], dim=0)

    gathered_dk = _all_gather_step_grads(step_dk, process_group)
    gathered_dv = _all_gather_step_grads(step_dv, process_group)
    dk = sum(gathered_dk[(kv_comm.rank + step) % kv_comm.world_size][step] for step in range(num_steps))
    dv = sum(gathered_dv[(kv_comm.rank + step) % kv_comm.world_size][step] for step in range(num_steps))

    return dq.to(q.dtype).unsqueeze(0), dk.to(q.dtype).unsqueeze(0), dv.to(q.dtype).unsqueeze(0)


def _zigzag_ring_flash_attn_varlen_backward_exact(
    process_group,
    dout,
    q,
    k,
    v,
    cu_seqlens,
    max_seqlen,
    half_index0,
    half_index1,
    softmax_scale,
    window_size,
):
    return zigzag_ring_flash_attn_varlen_backward_exact(
        process_group,
        dout,
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
        half_index0,
        half_index1,
        softmax_scale,
        window_size,
    )


def npu_forward(q,
                k,
                v,
                causal,
                cu_seqlens_q,
                cu_seqlens_kv,
                dropout_p,
                softmax_scale,
                deterministic,
                window_size,
                return_ctx: bool = False):
    outputs = _call_npu_fusion_attention(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        return_ctx=return_ctx,
    )
    ctx = None
    if return_ctx:
        outputs, ctx = outputs
    block_out, softmax_max, softmax_sum = outputs[:3]
    block_lse = softmax_max.to(torch.float32) + torch.log(softmax_sum.to(torch.float32))
    block_lse = reshape_npu_lse(block_lse, q.shape[0], q.shape[1])
    if return_ctx:
        return block_out, block_lse, ctx
    return block_out, block_lse


def _npu_forward(q,
                 k,
                 v,
                 causal,
                 cu_seqlens_q,
                 cu_seqlens_kv,
                 dropout_p,
                 softmax_scale,
                 deterministic,
                 window_size,
                 return_ctx: bool = False):
    return npu_forward(
        q,
        k,
        v,
        causal,
        cu_seqlens_q,
        cu_seqlens_kv,
        dropout_p,
        softmax_scale,
        deterministic,
        window_size,
        return_ctx=return_ctx,
    )


def npu_backward(dout, q, k, v, out, softmax_lse, causal, cu_seqlens_q, cu_seqlens_kv, dq_buffer, dk_buffer, dv_buffer, dropout_p,
                 softmax_scale, deterministic, window_size, backend_ctx=None, block_dlse=None):
    dq = dk = dv = None
    ctx = backend_ctx
    use_native_grad = ctx is not None and ctx.get('input_layout') in ('BSH', 'BNSD')
    if use_native_grad:
        grad_outputs = _call_npu_fusion_attention_grad(dout.to(q.dtype), q, k, v, ctx)
        dq, dk, dv = grad_outputs[:3]
    else:
        dq, dk, dv = manual_varlen_attention_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_kv,
            softmax_scale,
            causal,
            window_size,
        )

    dq_buffer.zero_()
    dk_buffer.zero_()
    dv_buffer.zero_()
    dq_buffer[:dq.shape[0]].copy_(dq.to(dq_buffer.dtype))
    dk_buffer[:dk.shape[0]].copy_(dk.to(dk_buffer.dtype))
    dv_buffer[:dv.shape[0]].copy_(dv.to(dv_buffer.dtype))
    if block_dlse is not None:
        dlse_dq, dlse_dk = manual_varlen_lse_backward(
            block_dlse,
            q,
            k,
            cu_seqlens_q,
            cu_seqlens_kv,
            softmax_scale,
            causal,
            window_size,
        )
        dq_buffer[:dlse_dq.shape[0]].add_(dlse_dq.to(dq_buffer.dtype))
        dk_buffer[:dlse_dk.shape[0]].add_(dlse_dk.to(dk_buffer.dtype))


def _npu_backward(dout, q, k, v, out, softmax_lse, causal, cu_seqlens_q, cu_seqlens_kv, dq_buffer, dk_buffer, dv_buffer, dropout_p,
                  softmax_scale, deterministic, window_size, backend_ctx=None, block_dlse=None):
    return npu_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        causal,
        cu_seqlens_q,
        cu_seqlens_kv,
        dq_buffer,
        dk_buffer,
        dv_buffer,
        dropout_p,
        softmax_scale,
        deterministic,
        window_size,
        backend_ctx=backend_ctx,
        block_dlse=block_dlse,
    )
