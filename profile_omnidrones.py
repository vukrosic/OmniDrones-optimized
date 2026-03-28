"""Profile OmniDrones core computational components on GPU.

Measures: GAE computation, policy forward/backward, observation normalization,
rotor dynamics, Lee position controller.
"""
import sys
import json
import time
from datetime import datetime

# Mock isaacsim
sys.modules["isaacsim"] = type(sys)("isaacsim")
sys.modules["isaacsim"].SimulationApp = None

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda")
torch.manual_seed(42)

results = {"baselines": {}, "improvements": [], "date": datetime.now().isoformat()}


def benchmark(fn, name, warmup=10, runs=50, **kwargs):
    """Benchmark a function with CUDA events for precise timing."""
    # Warmup
    for _ in range(warmup):
        fn(**kwargs)
    torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats()

        start.record()
        fn(**kwargs)
        end.record()
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))

    times = np.array(times)
    mem_mb = torch.cuda.max_memory_allocated() / 1e6

    result = {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "memory_mb": float(mem_mb),
        "runs": runs,
    }
    results["baselines"][name] = result
    print(f"  {name}: {result['mean_ms']:.3f} ± {result['std_ms']:.3f} ms (mem: {mem_mb:.1f} MB)")
    return result


# ============================================================
# 1. GAE (Generalized Advantage Estimation)
# ============================================================
print("\n=== 1. GAE Computation ===")

# Replicate the GAE logic from PPO
def compute_gae_original(rewards, values, dones, gamma=0.99, lmbda=0.95):
    """Standard sequential GAE as used in OmniDrones PPO."""
    T, N = rewards.shape[:2]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)
        else:
            next_value = values[t + 1]

        not_done = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last_gae = delta + gamma * lmbda * not_done * last_gae
        advantages[t] = last_gae

    return advantages

# Test sizes matching typical OmniDrones training
for label, T, N in [("small_32x64", 32, 64), ("medium_64x256", 64, 256), ("large_128x1024", 128, 1024)]:
    rewards = torch.randn(T, N, device=device)
    values = torch.randn(T, N, device=device)
    dones = torch.zeros(T, N, device=device)

    benchmark(
        compute_gae_original,
        f"gae_{label}",
        rewards=rewards, values=values, dones=dones,
    )


# ============================================================
# 2. Observation Normalization
# ============================================================
print("\n=== 2. Observation Normalization ===")

class RunningMeanStd:
    """Running mean/std as used in OmniDrones for obs normalization."""
    def __init__(self, shape, device):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = 1e-4

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / (torch.sqrt(self.var) + 1e-8)

    def update_and_normalize(self, x):
        self.update(x)
        return self.normalize(x)

for label, B, D in [("small_256x64", 256, 64), ("medium_1024x128", 1024, 128), ("large_4096x256", 4096, 256)]:
    rms = RunningMeanStd((D,), device)
    obs = torch.randn(B, D, device=device)

    benchmark(
        rms.update_and_normalize,
        f"obs_norm_{label}",
        x=obs,
    )


# ============================================================
# 3. Rotor Dynamics (from actuators/rotor_group.py)
# ============================================================
print("\n=== 3. Rotor Dynamics ===")

def rotor_dynamics(throttle_cmd, throttle_state, KF, KM, tau_up=0.43, tau_down=0.43, dt=0.01):
    """Simplified rotor dynamics from RotorGroup."""
    # First-order response
    alpha_up = dt / (tau_up + dt)
    alpha_down = dt / (tau_down + dt)
    alpha = torch.where(throttle_cmd > throttle_state, alpha_up, alpha_down)
    throttle_state = throttle_state + alpha * (throttle_cmd - throttle_state)

    # Thrust and moment
    thrust = KF * throttle_state ** 2
    moment = KM * throttle_state

    return throttle_state, thrust, moment

for label, N, R in [("16_drones_4rot", 16, 4), ("256_drones_4rot", 256, 4), ("1024_drones_4rot", 1024, 4)]:
    cmd = torch.rand(N, R, device=device)
    state = torch.rand(N, R, device=device) * 0.5
    KF = torch.tensor(1.0, device=device)
    KM = torch.tensor(0.01, device=device)

    benchmark(
        rotor_dynamics,
        f"rotor_{label}",
        throttle_cmd=cmd, throttle_state=state, KF=KF, KM=KM,
    )


# ============================================================
# 4. Lee Position Controller
# ============================================================
print("\n=== 4. Lee Position Controller (core math) ===")

def lee_controller_core(pos, quat, vel, angvel, target_pos, target_vel, gravity=-9.81):
    """Core math of Lee position controller."""
    # Position error -> desired acceleration
    Kp, Kd = 6.0, 4.0
    pos_err = pos - target_pos
    vel_err = vel - target_vel
    acc_des = -Kp * pos_err - Kd * vel_err
    acc_des[..., 2] -= gravity

    # Desired thrust (project onto body z-axis)
    # Simplified: just compute thrust magnitude
    thrust = torch.norm(acc_des, dim=-1, keepdim=True)

    # Attitude error (simplified)
    att_err = torch.cross(acc_des, torch.zeros_like(acc_des))

    return thrust, att_err

for label, N in [("16_drones", 16), ("256_drones", 256), ("1024_drones", 1024), ("4096_drones", 4096)]:
    pos = torch.randn(N, 3, device=device)
    quat = torch.randn(N, 4, device=device)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    vel = torch.randn(N, 3, device=device)
    angvel = torch.randn(N, 3, device=device)
    target_pos = torch.randn(N, 3, device=device)
    target_vel = torch.zeros(N, 3, device=device)

    benchmark(
        lee_controller_core,
        f"lee_ctrl_{label}",
        pos=pos, quat=quat, vel=vel, angvel=angvel,
        target_pos=target_pos, target_vel=target_vel,
    )


# ============================================================
# 5. Policy Forward Pass (MLP)
# ============================================================
print("\n=== 5. Policy Forward Pass ===")

from omni_drones.learning.modules.networks import MLP

for label, B, D_in, D_out in [("small_256x64", 256, 64, 8), ("medium_1024x128", 1024, 128, 16), ("large_4096x256", 4096, 256, 16)]:
    mlp = MLP([D_in, 256, 256, 256, D_out]).to(device)
    x = torch.randn(B, D_in, device=device)

    def fwd(model=mlp, inp=x):
        return model(inp)

    benchmark(fwd, f"mlp_fwd_{label}")

    # Also time backward
    def fwd_bwd(model=mlp, inp=x):
        out = model(inp)
        loss = out.sum()
        loss.backward()
        model.zero_grad()

    benchmark(fwd_bwd, f"mlp_fwd_bwd_{label}")


# ============================================================
# 6. Reward Computation (multi-component, typical OmniDrones pattern)
# ============================================================
print("\n=== 6. Multi-component Reward Computation ===")

def compute_reward(pos, target_pos, vel, angvel, effort):
    """Typical OmniDrones reward with multiple components."""
    # Position reward
    pos_err = torch.norm(pos - target_pos, dim=-1)
    pos_reward = 1.0 / (1.0 + pos_err ** 2)

    # Velocity penalty
    vel_penalty = -0.1 * torch.norm(vel, dim=-1)

    # Angular velocity penalty
    angvel_penalty = -0.05 * torch.norm(angvel, dim=-1)

    # Effort penalty
    effort_penalty = -0.01 * torch.norm(effort, dim=-1)

    total = pos_reward + vel_penalty + angvel_penalty + effort_penalty
    return total

for label, N in [("256_envs", 256), ("1024_envs", 1024), ("4096_envs", 4096)]:
    pos = torch.randn(N, 3, device=device)
    target_pos = torch.randn(N, 3, device=device)
    vel = torch.randn(N, 3, device=device)
    angvel = torch.randn(N, 3, device=device)
    effort = torch.randn(N, 4, device=device)

    benchmark(
        compute_reward,
        f"reward_{label}",
        pos=pos, target_pos=target_pos, vel=vel, angvel=angvel, effort=effort,
    )


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("PROFILING COMPLETE — Summary of bottlenecks:")
print("=" * 60)

sorted_results = sorted(results["baselines"].items(), key=lambda x: -x[1]["mean_ms"])
for name, r in sorted_results:
    print(f"  {r['mean_ms']:8.3f} ms  {name}")

# Save results
with open("improvement_log.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to improvement_log.json")
