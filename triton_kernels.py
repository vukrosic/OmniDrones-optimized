"""
Triton CUDA kernels for OmniDrones performance-critical operations.

Kernels:
1. GAE (Generalized Advantage Estimation) - parallel reverse scan
2. Observation normalization - fused update + normalize
3. Reward computation - fused multi-component reward
"""
import torch
import triton
import triton.language as tl


# ============================================================
# 1. GAE - Parallel Reverse Scan via Sequential-per-row Kernel
# ============================================================
# GAE has a sequential dependency along the time axis (reverse scan).
# We parallelize across the N (env) dimension, with each thread handling
# one environment's full time sequence. This eliminates the Python loop.

@triton.jit
def _gae_kernel(
    rewards_ptr, values_ptr, dones_ptr, advantages_ptr,
    T: tl.constexpr, N: tl.constexpr, stride_t: tl.constexpr, stride_n: tl.constexpr,
    gamma: tl.constexpr, lmbda: tl.constexpr,
):
    """Compute GAE for one environment (one column of the T x N matrix)."""
    env_id = tl.program_id(0)
    if env_id >= N:
        return

    last_gae = 0.0

    # Reverse scan through time
    for t_rev in range(T):
        t = T - 1 - t_rev

        offset = t * stride_t + env_id * stride_n
        reward = tl.load(rewards_ptr + offset)
        value = tl.load(values_ptr + offset)
        done = tl.load(dones_ptr + offset)

        not_done = 1.0 - done

        # Next value
        if t < T - 1:
            next_offset = (t + 1) * stride_t + env_id * stride_n
            next_value = tl.load(values_ptr + next_offset)
        else:
            next_value = 0.0

        delta = reward + gamma * next_value * not_done - value
        last_gae = delta + gamma * lmbda * not_done * last_gae

        tl.store(advantages_ptr + offset, last_gae)


def gae_triton(rewards, values, dones, gamma=0.99, lmbda=0.95):
    """Compute GAE using Triton kernel. Parallelizes across environments."""
    T, N = rewards.shape
    advantages = torch.empty_like(rewards)

    grid = (N,)
    _gae_kernel[grid](
        rewards, values, dones, advantages,
        T=T, N=N,
        stride_t=rewards.stride(0), stride_n=rewards.stride(1),
        gamma=gamma, lmbda=lmbda,
    )
    return advantages


# ============================================================
# 2. Fused Observation Normalization
# ============================================================
# Fuses: compute batch mean/var, update running stats, normalize - all in one kernel launch.

@triton.jit
def _obs_norm_kernel(
    obs_ptr, out_ptr, mean_ptr, var_ptr, count_ptr,
    B: tl.constexpr, D: tl.constexpr,
    stride_b: tl.constexpr, stride_d: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Normalize observation feature d across batch dimension."""
    d = tl.program_id(0)
    if d >= D:
        return

    # Compute batch mean and var for this feature
    batch_sum = 0.0
    batch_sq_sum = 0.0
    for b_start in range(0, B, BLOCK_B):
        b_offsets = b_start + tl.arange(0, BLOCK_B)
        mask = b_offsets < B
        vals = tl.load(obs_ptr + b_offsets * stride_b + d * stride_d, mask=mask, other=0.0)
        batch_sum += tl.sum(vals)
        batch_sq_sum += tl.sum(vals * vals)

    batch_mean = batch_sum / B
    batch_var = batch_sq_sum / B - batch_mean * batch_mean

    # Update running stats
    old_mean = tl.load(mean_ptr + d)
    old_var = tl.load(var_ptr + d)
    old_count = tl.load(count_ptr)

    new_count = old_count + B
    delta = batch_mean - old_mean
    new_mean = old_mean + delta * B / new_count
    m_a = old_var * old_count
    m_b = batch_var * B
    m2 = m_a + m_b + delta * delta * old_count * B / new_count
    new_var = m2 / new_count

    tl.store(mean_ptr + d, new_mean)
    tl.store(var_ptr + d, new_var)

    # Normalize output
    inv_std = 1.0 / tl.sqrt(new_var + 1e-8)
    for b_start in range(0, B, BLOCK_B):
        b_offsets = b_start + tl.arange(0, BLOCK_B)
        mask = b_offsets < B
        vals = tl.load(obs_ptr + b_offsets * stride_b + d * stride_d, mask=mask, other=0.0)
        normed = (vals - new_mean) * inv_std
        tl.store(out_ptr + b_offsets * stride_b + d * stride_d, normed, mask=mask)

    # Only thread 0 updates count
    if d == 0:
        tl.store(count_ptr, new_count)


class RunningMeanStdTriton:
    """Running mean/std with fused Triton update+normalize."""
    def __init__(self, shape, device):
        self.mean = torch.zeros(shape, device=device, dtype=torch.float32)
        self.var = torch.ones(shape, device=device, dtype=torch.float32)
        self.count = torch.tensor([1e-4], device=device, dtype=torch.float32)

    def update_and_normalize(self, x):
        B, D = x.shape
        out = torch.empty_like(x)
        BLOCK_B = min(triton.next_power_of_2(B), 1024)

        _obs_norm_kernel[(D,)](
            x, out, self.mean, self.var, self.count,
            B=B, D=D,
            stride_b=x.stride(0), stride_d=x.stride(1),
            BLOCK_B=BLOCK_B,
        )
        return out


# ============================================================
# 3. Fused Multi-component Reward
# ============================================================

@triton.jit
def _reward_kernel(
    pos_ptr, target_ptr, vel_ptr, angvel_ptr, effort_ptr, out_ptr,
    N: tl.constexpr,
    pos_stride: tl.constexpr,
    vel_stride: tl.constexpr,
    angvel_stride: tl.constexpr,
    effort_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fused reward: position + velocity + angvel + effort penalties."""
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N

    # Position error norm
    px = tl.load(pos_ptr + idx * pos_stride + 0, mask=mask, other=0.0)
    py = tl.load(pos_ptr + idx * pos_stride + 1, mask=mask, other=0.0)
    pz = tl.load(pos_ptr + idx * pos_stride + 2, mask=mask, other=0.0)
    tx = tl.load(target_ptr + idx * pos_stride + 0, mask=mask, other=0.0)
    ty = tl.load(target_ptr + idx * pos_stride + 1, mask=mask, other=0.0)
    tz = tl.load(target_ptr + idx * pos_stride + 2, mask=mask, other=0.0)

    dx, dy, dz = px - tx, py - ty, pz - tz
    pos_err_sq = dx * dx + dy * dy + dz * dz
    pos_reward = 1.0 / (1.0 + pos_err_sq)

    # Velocity penalty
    vx = tl.load(vel_ptr + idx * vel_stride + 0, mask=mask, other=0.0)
    vy = tl.load(vel_ptr + idx * vel_stride + 1, mask=mask, other=0.0)
    vz = tl.load(vel_ptr + idx * vel_stride + 2, mask=mask, other=0.0)
    vel_norm = tl.sqrt(vx * vx + vy * vy + vz * vz)
    vel_penalty = -0.1 * vel_norm

    # Angular velocity penalty
    ax = tl.load(angvel_ptr + idx * angvel_stride + 0, mask=mask, other=0.0)
    ay = tl.load(angvel_ptr + idx * angvel_stride + 1, mask=mask, other=0.0)
    az = tl.load(angvel_ptr + idx * angvel_stride + 2, mask=mask, other=0.0)
    angvel_norm = tl.sqrt(ax * ax + ay * ay + az * az)
    angvel_penalty = -0.05 * angvel_norm

    # Effort penalty
    e0 = tl.load(effort_ptr + idx * effort_stride + 0, mask=mask, other=0.0)
    e1 = tl.load(effort_ptr + idx * effort_stride + 1, mask=mask, other=0.0)
    e2 = tl.load(effort_ptr + idx * effort_stride + 2, mask=mask, other=0.0)
    e3 = tl.load(effort_ptr + idx * effort_stride + 3, mask=mask, other=0.0)
    effort_norm = tl.sqrt(e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3)
    effort_penalty = -0.01 * effort_norm

    total = pos_reward + vel_penalty + angvel_penalty + effort_penalty
    tl.store(out_ptr + idx, total, mask=mask)


def reward_triton(pos, target_pos, vel, angvel, effort):
    """Fused multi-component reward computation."""
    N = pos.shape[0]
    out = torch.empty(N, device=pos.device, dtype=pos.dtype)
    BLOCK = 256
    grid = ((N + BLOCK - 1) // BLOCK,)

    _reward_kernel[grid](
        pos, target_pos, vel, angvel, effort, out,
        N=N,
        pos_stride=pos.stride(0),
        vel_stride=vel.stride(0),
        angvel_stride=angvel.stride(0),
        effort_stride=effort.stride(0),
        BLOCK=BLOCK,
    )
    return out
