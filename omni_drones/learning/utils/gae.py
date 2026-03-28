# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _gae_nbt_kernel(
        reward_ptr, done_ptr, value_ptr, next_value_ptr, advantages_ptr,
        num_steps: tl.constexpr,
        stride_n: tl.constexpr,
        stride_t: tl.constexpr,
        stride_k: tl.constexpr,
        done_stride_n: tl.constexpr,
        done_stride_t: tl.constexpr,
        total_k: tl.constexpr,
        gamma: tl.constexpr,
        lmbda: tl.constexpr,
    ):
        """GAE kernel for (N, T, k) layout with single next_value (N, k).
        done is (N, T, 1) — broadcast across k."""
        pid = tl.program_id(0)
        n_idx = pid // total_k
        k_idx = pid % total_k

        nv = tl.load(next_value_ptr + n_idx * total_k + k_idx)

        gae = 0.0
        for step_rev in range(num_steps):
            step = num_steps - 1 - step_rev
            offset = n_idx * stride_n + step * stride_t + k_idx * stride_k
            done_offset = n_idx * done_stride_n + step * done_stride_t

            reward = tl.load(reward_ptr + offset)
            done = tl.load(done_ptr + done_offset)
            value = tl.load(value_ptr + offset)

            not_done = 1.0 - done
            delta = reward + gamma * nv * not_done - value
            gae = delta + gamma * lmbda * not_done * gae

            tl.store(advantages_ptr + offset, gae)
            nv = value

    @triton.jit
    def _gae_tnk_kernel(
        reward_ptr, done_ptr, value_ptr, next_value_ptr, advantages_ptr,
        num_steps: tl.constexpr,
        stride_t: tl.constexpr,
        stride_n: tl.constexpr,
        stride_k: tl.constexpr,
        done_stride_t: tl.constexpr,
        done_stride_n: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        gamma: tl.constexpr,
        lmbda: tl.constexpr,
    ):
        """GAE kernel for (T, N, k) layout with single next_value (N, k).
        done is (T, N, 1) — broadcast across k."""
        pid = tl.program_id(0)
        n_idx = pid // K
        k_idx = pid % K

        nv = tl.load(next_value_ptr + n_idx * K + k_idx)

        gae = 0.0
        for step_rev in range(num_steps):
            step = num_steps - 1 - step_rev
            offset = step * stride_t + n_idx * stride_n + k_idx * stride_k
            done_offset = step * done_stride_t + n_idx * done_stride_n

            reward = tl.load(reward_ptr + offset)
            done = tl.load(done_ptr + done_offset)
            value = tl.load(value_ptr + offset)

            not_done = 1.0 - done
            delta = reward + gamma * nv * not_done - value
            gae = delta + gamma * lmbda * not_done * gae

            tl.store(advantages_ptr + offset, gae)
            nv = value


def compute_gae(
    reward: torch.Tensor,  # [N, T, k]
    done: torch.Tensor,  # [N, T, 1]
    value: torch.Tensor,  # [N, T, k]
    next_value: torch.Tensor,  # [N, k]
    gamma=0.99,
    lmbda=0.95,
):
    assert reward.shape == value.shape

    if HAS_TRITON and reward.is_cuda:
        return _compute_gae_triton(reward, done, value, next_value, gamma, lmbda)
    return _compute_gae_python(reward, done, value, next_value, gamma, lmbda)


def _compute_gae_triton(reward, done, value, next_value, gamma, lmbda):
    N, T, k = reward.shape
    advantages = torch.empty_like(reward)

    reward = reward.contiguous()
    done = done.contiguous().float()
    value = value.contiguous()
    next_value = next_value.contiguous()

    grid = (N * k,)
    _gae_nbt_kernel[grid](
        reward, done, value, next_value, advantages,
        num_steps=T,
        stride_n=reward.stride(0),
        stride_t=reward.stride(1),
        stride_k=reward.stride(2),
        done_stride_n=done.stride(0),
        done_stride_t=done.stride(1),
        total_k=k,
        gamma=gamma,
        lmbda=lmbda,
    )
    returns = advantages + value
    return advantages, returns


def _compute_gae_python(reward, done, value, next_value, gamma, lmbda):
    not_done = 1.0 - done.float()
    batch_size, num_steps = not_done.shape[:2]
    gae = 0
    advantages = torch.zeros_like(reward)
    for step in reversed(range(num_steps)):
        delta = (
            reward[:, step]
            + gamma * next_value * not_done[:, step]
            - value[:, step]
        )
        advantages[:, step] = gae = delta + (gamma * lmbda * not_done[:, step] * gae)
        next_value = value[:, step]
    returns = advantages + value
    return advantages, returns


def compute_gae_(
    reward: torch.Tensor,  # [T, N, k]
    done: torch.Tensor,  # [T, N, 1]
    value: torch.Tensor,  # [T, N, k]
    next_value: torch.Tensor,  # [N, k]
    gamma=0.99,
    lmbda=0.95,
):
    assert reward.shape == value.shape

    if HAS_TRITON and reward.is_cuda:
        return _compute_gae_triton_(reward, done, value, next_value, gamma, lmbda)
    return _compute_gae_python_(reward, done, value, next_value, gamma, lmbda)


def _compute_gae_triton_(reward, done, value, next_value, gamma, lmbda):
    T, N, k = reward.shape
    advantages = torch.empty_like(reward)

    reward = reward.contiguous()
    done = done.contiguous().float()
    value = value.contiguous()
    next_value = next_value.contiguous()

    grid = (N * k,)
    _gae_tnk_kernel[grid](
        reward, done, value, next_value, advantages,
        num_steps=T,
        stride_t=reward.stride(0),
        stride_n=reward.stride(1),
        stride_k=reward.stride(2),
        done_stride_t=done.stride(0),
        done_stride_n=done.stride(1),
        N=N,
        K=k,
        gamma=gamma,
        lmbda=lmbda,
    )
    returns = advantages + value
    return advantages, returns


def _compute_gae_python_(reward, done, value, next_value, gamma, lmbda):
    not_done = 1.0 - done.float()
    num_steps = not_done.shape[0]
    gae = 0
    advantages = torch.zeros_like(reward)
    for step in reversed(range(num_steps)):
        delta = reward[step] + gamma * next_value * not_done[step] - value[step]
        advantages[step] = gae = delta + (gamma * lmbda * not_done[step] * gae)
        next_value = value[step]
    returns = advantages + value
    return advantages, returns
