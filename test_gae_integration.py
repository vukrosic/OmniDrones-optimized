"""Test Triton GAE against original OmniDrones GAE with exact same interface."""
import sys
import torch
import numpy as np

sys.modules["isaacsim"] = type(sys)("isaacsim")
sys.modules["isaacsim"].SimulationApp = None

device = torch.device("cuda")

# Import original
from omni_drones.learning.ppo.common import GAE as OriginalGAE

# Import Triton version
sys.path.insert(0, "/workspace/OmniDrones")
from gae_triton_integrated import GAE as TritonGAE

print("=" * 60)
print("CORRECTNESS TEST — Triton GAE vs Original GAE")
print("=" * 60)

all_pass = True
for batch, steps, agents in [(4, 32, 1), (16, 32, 4), (64, 64, 4), (256, 128, 1)]:
    # Shape: (batch, steps, agents, 1) — matching OmniDrones
    reward = torch.randn(batch, steps, agents, 1, device=device)
    terminated = (torch.rand(batch, steps, agents, 1, device=device) > 0.95).float()
    value = torch.randn(batch, steps, agents, 1, device=device)
    next_value = torch.randn(batch, steps, agents, 1, device=device)

    orig_gae = OriginalGAE(0.99, 0.95).to(device)
    triton_gae = TritonGAE(0.99, 0.95).to(device)

    adv_orig, ret_orig = orig_gae(reward, terminated, value, next_value)
    adv_triton, ret_triton = triton_gae(reward, terminated, value, next_value)

    adv_diff = (adv_orig - adv_triton).abs().max().item()
    ret_diff = (ret_orig - ret_triton).abs().max().item()
    adv_close = torch.allclose(adv_orig, adv_triton, rtol=1e-4, atol=1e-4)
    ret_close = torch.allclose(ret_orig, ret_triton, rtol=1e-4, atol=1e-4)

    status = "PASS" if (adv_close and ret_close) else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"  ({batch},{steps},{agents},1): adv_diff={adv_diff:.2e} ret_diff={ret_diff:.2e} [{status}]")

print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

# Performance comparison
print("\n" + "=" * 60)
print("PERFORMANCE — Triton GAE vs Original GAE")
print("=" * 60)

def benchmark(fn, warmup=10, runs=50, **kwargs):
    for _ in range(warmup):
        fn(**kwargs)
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(**kwargs)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return np.mean(times), np.std(times)

for batch, steps, agents in [(16, 32, 4), (64, 64, 4), (256, 128, 1), (64, 32, 8)]:
    reward = torch.randn(batch, steps, agents, 1, device=device)
    terminated = torch.zeros(batch, steps, agents, 1, device=device)
    value = torch.randn(batch, steps, agents, 1, device=device)
    next_value = torch.randn(batch, steps, agents, 1, device=device)

    orig_gae = OriginalGAE(0.99, 0.95).to(device)
    triton_gae = TritonGAE(0.99, 0.95).to(device)

    orig_mean, orig_std = benchmark(orig_gae, reward=reward, terminated=terminated, value=value, next_value=next_value)
    triton_mean, triton_std = benchmark(triton_gae, reward=reward, terminated=terminated, value=value, next_value=next_value)

    speedup = orig_mean / triton_mean
    print(f"  ({batch},{steps},{agents},1): orig={orig_mean:.3f}ms  triton={triton_mean:.3f}ms  speedup={speedup:.1f}x")
