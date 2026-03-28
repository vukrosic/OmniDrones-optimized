"""
Benchmark and verify correctness of Triton kernels vs. original implementations.
"""
import sys
import json
import time
from datetime import datetime

sys.modules["isaacsim"] = type(sys)("isaacsim")
sys.modules["isaacsim"].SimulationApp = None

import torch
import numpy as np

device = torch.device("cuda")
torch.manual_seed(42)

from triton_kernels import gae_triton, RunningMeanStdTriton, reward_triton


def benchmark(fn, name, warmup=10, runs=50, **kwargs):
    for _ in range(warmup):
        fn(**kwargs)
    torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
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
        "memory_mb": float(mem_mb),
    }
    print(f"  {name}: {result['mean_ms']:.3f} +/- {result['std_ms']:.3f} ms")
    return result


# ============================================================
# Original implementations for comparison
# ============================================================

def compute_gae_original(rewards, values, dones, gamma=0.99, lmbda=0.95):
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


class RunningMeanStdOriginal:
    def __init__(self, shape, device):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = 1e-4

    def update_and_normalize(self, x):
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
        return (x - self.mean) / (torch.sqrt(self.var) + 1e-8)


def compute_reward_original(pos, target_pos, vel, angvel, effort):
    pos_err = torch.norm(pos - target_pos, dim=-1)
    pos_reward = 1.0 / (1.0 + pos_err ** 2)
    vel_penalty = -0.1 * torch.norm(vel, dim=-1)
    angvel_penalty = -0.05 * torch.norm(angvel, dim=-1)
    effort_penalty = -0.01 * torch.norm(effort, dim=-1)
    return pos_reward + vel_penalty + angvel_penalty + effort_penalty


# ============================================================
# Correctness Verification
# ============================================================
print("=" * 60)
print("CORRECTNESS VERIFICATION")
print("=" * 60)

# 1. GAE
print("\n--- GAE ---")
for T, N in [(32, 64), (64, 256), (128, 1024)]:
    rewards = torch.randn(T, N, device=device)
    values = torch.randn(T, N, device=device)
    dones = (torch.rand(T, N, device=device) > 0.95).float()

    ref = compute_gae_original(rewards, values, dones)
    out = gae_triton(rewards, values, dones)

    max_diff = (ref - out).abs().max().item()
    allclose = torch.allclose(ref, out, rtol=1e-4, atol=1e-4)
    print(f"  T={T}, N={N}: max_diff={max_diff:.2e}, allclose={allclose}")
    if not allclose:
        print(f"    WARNING: Results differ! ref[:3,0]={ref[:3,0].tolist()}, out[:3,0]={out[:3,0].tolist()}")

# 2. Obs Normalization
print("\n--- Observation Normalization ---")
for B, D in [(256, 64), (1024, 128), (4096, 256)]:
    obs = torch.randn(B, D, device=device)

    rms_orig = RunningMeanStdOriginal((D,), device)
    rms_triton = RunningMeanStdTriton((D,), device)

    ref = rms_orig.update_and_normalize(obs)
    out = rms_triton.update_and_normalize(obs)

    max_diff = (ref - out).abs().max().item()
    allclose = torch.allclose(ref, out, rtol=1e-3, atol=1e-3)
    print(f"  B={B}, D={D}: max_diff={max_diff:.2e}, allclose={allclose}")

# 3. Reward
print("\n--- Reward Computation ---")
for N in [256, 1024, 4096]:
    pos = torch.randn(N, 3, device=device)
    target_pos = torch.randn(N, 3, device=device)
    vel = torch.randn(N, 3, device=device)
    angvel = torch.randn(N, 3, device=device)
    effort = torch.randn(N, 4, device=device)

    ref = compute_reward_original(pos, target_pos, vel, angvel, effort)
    out = reward_triton(pos, target_pos, vel, angvel, effort)

    max_diff = (ref - out).abs().max().item()
    allclose = torch.allclose(ref, out, rtol=1e-4, atol=1e-4)
    print(f"  N={N}: max_diff={max_diff:.2e}, allclose={allclose}")

# ============================================================
# Performance Benchmarks
# ============================================================
print("\n" + "=" * 60)
print("PERFORMANCE BENCHMARKS")
print("=" * 60)

improvements = []

# 1. GAE
print("\n--- GAE ---")
for label, T, N in [("small_32x64", 32, 64), ("medium_64x256", 64, 256), ("large_128x1024", 128, 1024)]:
    rewards = torch.randn(T, N, device=device)
    values = torch.randn(T, N, device=device)
    dones = torch.zeros(T, N, device=device)

    orig = benchmark(compute_gae_original, f"gae_orig_{label}", rewards=rewards, values=values, dones=dones)
    triton_r = benchmark(gae_triton, f"gae_triton_{label}", rewards=rewards, values=values, dones=dones)

    speedup = orig["mean_ms"] / triton_r["mean_ms"]
    print(f"  => Speedup: {speedup:.2f}x\n")
    improvements.append({
        "change": f"GAE Triton kernel ({label})",
        "before_ms": orig["mean_ms"],
        "after_ms": triton_r["mean_ms"],
        "speedup": f"{speedup:.2f}x",
    })

# 2. Obs Normalization
print("\n--- Observation Normalization ---")
for label, B, D in [("small_256x64", 256, 64), ("medium_1024x128", 1024, 128), ("large_4096x256", 4096, 256)]:
    obs = torch.randn(B, D, device=device)

    rms_orig = RunningMeanStdOriginal((D,), device)
    rms_triton_obj = RunningMeanStdTriton((D,), device)

    orig = benchmark(rms_orig.update_and_normalize, f"obs_orig_{label}", x=obs)
    triton_r = benchmark(rms_triton_obj.update_and_normalize, f"obs_triton_{label}", x=obs)

    speedup = orig["mean_ms"] / triton_r["mean_ms"]
    print(f"  => Speedup: {speedup:.2f}x\n")
    improvements.append({
        "change": f"Obs norm Triton kernel ({label})",
        "before_ms": orig["mean_ms"],
        "after_ms": triton_r["mean_ms"],
        "speedup": f"{speedup:.2f}x",
    })

# 3. Reward
print("\n--- Reward Computation ---")
for label, N in [("256_envs", 256), ("1024_envs", 1024), ("4096_envs", 4096)]:
    pos = torch.randn(N, 3, device=device)
    target_pos = torch.randn(N, 3, device=device)
    vel = torch.randn(N, 3, device=device)
    angvel = torch.randn(N, 3, device=device)
    effort = torch.randn(N, 4, device=device)

    orig = benchmark(compute_reward_original, f"reward_orig_{label}", pos=pos, target_pos=target_pos, vel=vel, angvel=angvel, effort=effort)
    triton_r = benchmark(reward_triton, f"reward_triton_{label}", pos=pos, target_pos=target_pos, vel=vel, angvel=angvel, effort=effort)

    speedup = orig["mean_ms"] / triton_r["mean_ms"]
    print(f"  => Speedup: {speedup:.2f}x\n")
    improvements.append({
        "change": f"Reward Triton kernel ({label})",
        "before_ms": orig["mean_ms"],
        "after_ms": triton_r["mean_ms"],
        "speedup": f"{speedup:.2f}x",
    })

# Load existing log and append
try:
    with open("improvement_log.json") as f:
        log = json.load(f)
except FileNotFoundError:
    log = {"baselines": {}, "improvements": []}

log["improvements"].extend(improvements)
log["last_benchmark_date"] = datetime.now().isoformat()

with open("improvement_log.json", "w") as f:
    json.dump(log, f, indent=2)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for imp in improvements:
    print(f"  {imp['change']}: {imp['before_ms']:.3f} -> {imp['after_ms']:.3f} ms ({imp['speedup']})")
print(f"\nResults saved to improvement_log.json")
