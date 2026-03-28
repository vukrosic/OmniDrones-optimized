"""
Drop-in replacement for OmniDrones GAE with Triton kernel.
Replaces omni_drones/learning/ppo/common.py GAE class.
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gae_kernel(
    reward_ptr, terminated_ptr, value_ptr, next_value_ptr, advantages_ptr,
    num_steps: tl.constexpr,
    stride_batch: tl.constexpr,
    stride_step: tl.constexpr,
    stride_tail: tl.constexpr,
    total_tail: tl.constexpr,
    gamma: tl.constexpr,
    lmbda: tl.constexpr,
):
    """Compute GAE for one (batch, tail_element) pair.

    Tensors are shaped (batch, steps, *tail) where tail can be (agents, 1) etc.
    Each program handles one element across the full time dimension.
    """
    pid = tl.program_id(0)
    batch_idx = pid // total_tail
    tail_idx = pid % total_tail

    gae = 0.0
    for step_rev in range(num_steps):
        step = num_steps - 1 - step_rev
        offset = batch_idx * stride_batch + step * stride_step + tail_idx * stride_tail

        reward = tl.load(reward_ptr + offset)
        terminated = tl.load(terminated_ptr + offset)
        value = tl.load(value_ptr + offset)
        next_val = tl.load(next_value_ptr + offset)

        not_done = 1.0 - terminated
        delta = reward + gamma * next_val * not_done - value
        gae = delta + gamma * lmbda * not_done * gae

        tl.store(advantages_ptr + offset, gae)


class GAE(nn.Module):
    """Drop-in replacement for OmniDrones GAE using Triton kernel.

    Falls back to Python loop if Triton is unavailable or tensors are on CPU.
    """
    def __init__(self, gamma, lmbda):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.gamma: torch.Tensor
        self.lmbda: torch.Tensor

    def forward(
        self,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        value: torch.Tensor,
        next_value: torch.Tensor,
    ):
        if reward.is_cuda:
            return self._forward_triton(reward, terminated, value, next_value)
        return self._forward_python(reward, terminated, value, next_value)

    def _forward_triton(self, reward, terminated, value, next_value):
        # Shape: (batch, num_steps, *tail)
        batch_size = reward.shape[0]
        num_steps = reward.shape[1]
        # Flatten trailing dims
        tail_shape = reward.shape[2:]
        total_tail = 1
        for s in tail_shape:
            total_tail *= s

        advantages = torch.empty_like(reward)

        # Ensure contiguous
        reward = reward.contiguous()
        terminated = terminated.contiguous().float()
        value = value.contiguous()
        next_value = next_value.contiguous()

        total_programs = batch_size * total_tail
        grid = (total_programs,)

        stride_batch = reward.stride(0)
        stride_step = reward.stride(1)
        # stride for the flattened tail dimension
        stride_tail = 1  # contiguous, so last dim stride is 1

        _gae_kernel[grid](
            reward, terminated, value, next_value, advantages,
            num_steps=num_steps,
            stride_batch=stride_batch,
            stride_step=stride_step,
            stride_tail=stride_tail,
            total_tail=total_tail,
            gamma=float(self.gamma),
            lmbda=float(self.lmbda),
        )

        returns = advantages + value
        return advantages, returns

    def _forward_python(self, reward, terminated, value, next_value):
        """Original Python fallback."""
        num_steps = terminated.shape[1]
        advantages = torch.zeros_like(reward)
        not_done = 1 - terminated.float()
        gae = 0
        for step in reversed(range(num_steps)):
            delta = (
                reward[:, step]
                + self.gamma * next_value[:, step] * not_done[:, step]
                - value[:, step]
            )
            advantages[:, step] = gae = delta + (self.gamma * self.lmbda * not_done[:, step] * gae)
        returns = advantages + value
        return advantages, returns
