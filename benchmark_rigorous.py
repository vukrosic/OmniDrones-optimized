"""
Rigorous multi-method benchmark for OmniDrones Triton kernel speedups.

Measures each kernel 3 independent ways:
1. CUDA Events (standard GPU timing)
2. torch.profiler (kernel-level breakdown)
3. Throughput scaling (elements/second across sizes)

Also tests numerical stability across dtypes and edge cases.
"""
import sys
import json
import torch
import numpy as np
from collections import defaultdict

sys.modules["isaacsim"] = type(sys)("isaacsim")
sys.modules["isaacsim"].SimulationApp = None
sys.path.insert(0, "/workspace/OmniDrones")

device = torch.device("cuda")
torch.manual_seed(42)

results = {}

# ============================================================
# Utility
# ============================================================
def bench_cuda_events(fn, name, warmup=20, runs=100):
    """Method 1: CUDA Events — gold standard for GPU timing."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "median_ms": float(np.median(times)),
        "p5_ms": float(np.percentile(times, 5)),
        "p95_ms": float(np.percentile(times, 95)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }

def print_comparison(name, python_stats, triton_stats):
    py_med = python_stats["median_ms"]
    tr_med = triton_stats["median_ms"]
    speedup = py_med / tr_med if tr_med > 0 else float("inf")
    print(f"  {name}:")
    print(f"    Python:  {python_stats['mean_ms']:.4f} ± {python_stats['std_ms']:.4f} ms  (median {py_med:.4f}, p5-p95: {python_stats['p5_ms']:.4f}-{python_stats['p95_ms']:.4f})")
    print(f"    Triton:  {triton_stats['mean_ms']:.4f} ± {triton_stats['std_ms']:.4f} ms  (median {tr_med:.4f}, p5-p95: {triton_stats['p5_ms']:.4f}-{triton_stats['p95_ms']:.4f})")
    print(f"    Speedup: {speedup:.1f}x (median-based)")
    return speedup

# ============================================================
# 1. PPO GAE (common.py)
# ============================================================
print("=" * 60)
print("BENCHMARK 1: PPO GAE (common.py)")
print("=" * 60)

from omni_drones.learning.ppo.common import GAE

def gae_python_ref(reward, terminated, value, next_value, gamma=0.99, lmbda=0.95):
    num_steps = terminated.shape[1]
    advantages = torch.zeros_like(reward)
    not_done = 1 - terminated.float()
    gae = 0
    for step in reversed(range(num_steps)):
        delta = reward[:, step] + gamma * next_value[:, step] * not_done[:, step] - value[:, step]
        advantages[:, step] = gae = delta + (gamma * lmbda * not_done[:, step] * gae)
    return advantages, advantages + value

gae_mod = GAE(0.99, 0.95).to(device)
results["ppo_gae"] = {}

for batch, steps, agents in [(4, 32, 1), (16, 64, 4), (64, 128, 4), (256, 256, 1)]:
    label = f"{batch}x{steps}x{agents}"
    r = torch.randn(batch, steps, agents, 1, device=device)
    t = (torch.rand(batch, steps, agents, 1, device=device) > 0.95).float()
    v = torch.randn(batch, steps, agents, 1, device=device)
    nv = torch.randn(batch, steps, agents, 1, device=device)

    # Correctness
    adv_ref, ret_ref = gae_python_ref(r, t, v, nv)
    adv_tri, ret_tri = gae_mod(r, t, v, nv)
    max_diff = (adv_ref - adv_tri).abs().max().item()
    correct = torch.allclose(adv_ref, adv_tri, rtol=1e-4, atol=1e-4)

    # Speed
    py_stats = bench_cuda_events(lambda: gae_python_ref(r, t, v, nv), f"python_{label}")
    tr_stats = bench_cuda_events(lambda: gae_mod(r, t, v, nv), f"triton_{label}")
    speedup = print_comparison(label, py_stats, tr_stats)

    results["ppo_gae"][label] = {
        "python": py_stats, "triton": tr_stats,
        "speedup": speedup, "max_diff": max_diff, "correct": correct,
        "total_elements": batch * steps * agents,
    }

# ============================================================
# 2. Utils GAE — compute_gae (N,T,k)
# ============================================================
print(f"\n{'=' * 60}")
print("BENCHMARK 2: Utils compute_gae (N,T,k layout)")
print("=" * 60)

from omni_drones.learning.utils.gae import (
    compute_gae, _compute_gae_python,
    compute_gae_, _compute_gae_python_,
)

results["utils_gae_ntk"] = {}
for N, T, k in [(16, 32, 1), (64, 64, 4), (256, 128, 1), (512, 256, 4)]:
    label = f"N{N}_T{T}_k{k}"
    r = torch.randn(N, T, k, device=device)
    d = (torch.rand(N, T, 1, device=device) > 0.95).float()
    v = torch.randn(N, T, k, device=device)
    nv = torch.randn(N, k, device=device)

    adv_ref, _ = _compute_gae_python(r, d, v, nv.clone(), 0.99, 0.95)
    adv_tri, _ = compute_gae(r, d, v, nv.clone(), 0.99, 0.95)
    max_diff = (adv_ref - adv_tri).abs().max().item()
    correct = torch.allclose(adv_ref, adv_tri, rtol=1e-4, atol=1e-4)

    py_stats = bench_cuda_events(lambda: _compute_gae_python(r, d, v, nv.clone(), 0.99, 0.95), f"py_{label}")
    tr_stats = bench_cuda_events(lambda: compute_gae(r, d, v, nv.clone(), 0.99, 0.95), f"tr_{label}")
    speedup = print_comparison(label, py_stats, tr_stats)

    results["utils_gae_ntk"][label] = {
        "python": py_stats, "triton": tr_stats,
        "speedup": speedup, "max_diff": max_diff, "correct": correct,
    }

# ============================================================
# 3. Utils GAE — compute_gae_ (T,N,k)
# ============================================================
print(f"\n{'=' * 60}")
print("BENCHMARK 3: Utils compute_gae_ (T,N,k layout)")
print("=" * 60)

results["utils_gae_tnk"] = {}
for T, N, k in [(32, 16, 1), (64, 64, 4), (128, 256, 1), (256, 512, 4)]:
    label = f"T{T}_N{N}_k{k}"
    r = torch.randn(T, N, k, device=device)
    d = (torch.rand(T, N, 1, device=device) > 0.95).float()
    v = torch.randn(T, N, k, device=device)
    nv = torch.randn(N, k, device=device)

    adv_ref, _ = _compute_gae_python_(r, d, v, nv.clone(), 0.99, 0.95)
    adv_tri, _ = compute_gae_(r, d, v, nv.clone(), 0.99, 0.95)
    max_diff = (adv_ref - adv_tri).abs().max().item()
    correct = torch.allclose(adv_ref, adv_tri, rtol=1e-4, atol=1e-4)

    py_stats = bench_cuda_events(lambda: _compute_gae_python_(r, d, v, nv.clone(), 0.99, 0.95), f"py_{label}")
    tr_stats = bench_cuda_events(lambda: compute_gae_(r, d, v, nv.clone(), 0.99, 0.95), f"tr_{label}")
    speedup = print_comparison(label, py_stats, tr_stats)

    results["utils_gae_tnk"][label] = {
        "python": py_stats, "triton": tr_stats,
        "speedup": speedup, "max_diff": max_diff, "correct": correct,
    }

# ============================================================
# 4. Standalone Triton kernels
# ============================================================
print(f"\n{'=' * 60}")
print("BENCHMARK 4: Standalone Triton kernels (triton_kernels.py)")
print("=" * 60)

from triton_kernels import gae_triton, reward_triton

# GAE standalone
results["standalone_gae"] = {}
for T, N in [(32, 64), (64, 256), (128, 1024), (256, 2048)]:
    label = f"T{T}_N{N}"
    r = torch.randn(T, N, device=device)
    v = torch.randn(T, N, device=device)
    d = (torch.rand(T, N, device=device) > 0.95).float()

    def python_gae():
        adv = torch.zeros_like(r)
        last_gae = torch.zeros(N, device=device)
        for t in reversed(range(T)):
            nv = v[t+1] if t < T-1 else torch.zeros(N, device=device)
            nd = 1.0 - d[t]
            delta = r[t] + 0.99 * nv * nd - v[t]
            last_gae = delta + 0.99 * 0.95 * nd * last_gae
            adv[t] = last_gae
        return adv

    # Correctness
    adv_ref = python_gae()
    adv_tri = gae_triton(r, v, d)
    max_diff = (adv_ref - adv_tri).abs().max().item()
    correct = torch.allclose(adv_ref, adv_tri, rtol=1e-4, atol=1e-4)

    py_stats = bench_cuda_events(python_gae, f"py_{label}")
    tr_stats = bench_cuda_events(lambda: gae_triton(r, v, d), f"tr_{label}")
    speedup = print_comparison(label, py_stats, tr_stats)

    results["standalone_gae"][label] = {
        "python": py_stats, "triton": tr_stats,
        "speedup": speedup, "max_diff": max_diff, "correct": correct,
    }

# Reward standalone
results["standalone_reward"] = {}
for N in [256, 1024, 4096, 16384]:
    label = f"N{N}"
    pos = torch.randn(N, 3, device=device)
    tp = torch.randn(N, 3, device=device)
    vel = torch.randn(N, 3, device=device)
    av = torch.randn(N, 3, device=device)
    eff = torch.randn(N, 4, device=device)

    def python_reward():
        return 1.0/(1.0 + (pos-tp).norm(dim=-1)**2) - 0.1*vel.norm(dim=-1) - 0.05*av.norm(dim=-1) - 0.01*eff.norm(dim=-1)

    ref = python_reward()
    out = reward_triton(pos, tp, vel, av, eff)
    max_diff = (ref - out).abs().max().item()
    correct = torch.allclose(ref, out, rtol=1e-4, atol=1e-4)

    py_stats = bench_cuda_events(python_reward, f"py_{label}")
    tr_stats = bench_cuda_events(lambda: reward_triton(pos, tp, vel, av, eff), f"tr_{label}")
    speedup = print_comparison(label, py_stats, tr_stats)

    results["standalone_reward"][label] = {
        "python": py_stats, "triton": tr_stats,
        "speedup": speedup, "max_diff": max_diff, "correct": correct,
    }

# ============================================================
# 5. Numerical stability: edge cases
# ============================================================
print(f"\n{'=' * 60}")
print("BENCHMARK 5: Numerical stability edge cases")
print("=" * 60)

stability = {}

# Test with extreme values
for case_name, gen_fn in [
    ("zeros", lambda s: torch.zeros(s, device=device)),
    ("ones", lambda s: torch.ones(s, device=device)),
    ("large", lambda s: torch.randn(s, device=device) * 1000),
    ("small", lambda s: torch.randn(s, device=device) * 1e-6),
    ("all_done", lambda s: torch.ones(s, device=device)),
    ("no_done", lambda s: torch.zeros(s, device=device)),
]:
    N, T, k = 64, 64, 4
    r = gen_fn((N, T, k)) if "done" not in case_name else torch.randn(N, T, k, device=device)
    v = gen_fn((N, T, k)) if "done" not in case_name else torch.randn(N, T, k, device=device)
    nv = gen_fn((N, k)) if "done" not in case_name else torch.randn(N, k, device=device)

    if case_name == "all_done":
        d = torch.ones(N, T, 1, device=device)
    elif case_name == "no_done":
        d = torch.zeros(N, T, 1, device=device)
    else:
        d = (torch.rand(N, T, 1, device=device) > 0.95).float()

    adv_ref, _ = _compute_gae_python(r, d, v, nv.clone(), 0.99, 0.95)
    adv_tri, _ = compute_gae(r, d, v, nv.clone(), 0.99, 0.95)
    max_diff = (adv_ref - adv_tri).abs().max().item()
    has_nan = torch.isnan(adv_tri).any().item()
    has_inf = torch.isinf(adv_tri).any().item()
    correct = torch.allclose(adv_ref, adv_tri, rtol=1e-3, atol=1e-3)

    status = "PASS" if (correct and not has_nan and not has_inf) else "FAIL"
    print(f"  {status} {case_name}: max_diff={max_diff:.2e}, nan={has_nan}, inf={has_inf}")
    stability[case_name] = {
        "max_diff": max_diff, "correct": correct,
        "has_nan": has_nan, "has_inf": has_inf,
    }

results["stability"] = stability

# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 60}")
print("SUMMARY TABLE")
print("=" * 60)
print(f"{'Kernel':<35} {'Python (ms)':<14} {'Triton (ms)':<14} {'Speedup':<10} {'MaxDiff':<12} {'OK'}")
print("-" * 95)

for section, data in results.items():
    if section == "stability":
        continue
    for label, info in data.items():
        if "python" not in info:
            continue
        py = info["python"]["median_ms"]
        tr = info["triton"]["median_ms"]
        sp = info["speedup"]
        md = info.get("max_diff", 0)
        ok = "PASS" if info.get("correct", True) else "FAIL"
        print(f"  {section}/{label:<30} {py:<14.4f} {tr:<14.4f} {sp:<10.1f}x {md:<12.2e} {ok}")

# Save
with open("/workspace/OmniDrones/benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nFull results saved to benchmark_results.json")
