"""Test the patched utils/gae.py against original behavior."""
import sys
import torch
import numpy as np

sys.modules["isaacsim"] = type(sys)("isaacsim")
sys.modules["isaacsim"].SimulationApp = None

device = torch.device("cuda")
torch.manual_seed(42)

# Original implementations for reference
def compute_gae_ref(reward, done, value, next_value, gamma=0.99, lmbda=0.95):
    not_done = 1.0 - done.float()
    batch_size, num_steps = not_done.shape[:2]
    gae = 0
    advantages = torch.zeros_like(reward)
    for step in reversed(range(num_steps)):
        delta = reward[:, step] + gamma * next_value * not_done[:, step] - value[:, step]
        advantages[:, step] = gae = delta + (gamma * lmbda * not_done[:, step] * gae)
        next_value = value[:, step]
    returns = advantages + value
    return advantages, returns

def compute_gae_ref_(reward, done, value, next_value, gamma=0.99, lmbda=0.95):
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

# Import patched versions
from omni_drones.learning.utils.gae import compute_gae, compute_gae_

print("=== Testing compute_gae (N, T, k) ===")
for N, T, k in [(16, 32, 1), (64, 64, 4), (256, 128, 1)]:
    reward = torch.randn(N, T, k, device=device)
    done = (torch.rand(N, T, 1, device=device) > 0.95).float()
    value = torch.randn(N, T, k, device=device)
    next_value = torch.randn(N, k, device=device)

    adv_ref, ret_ref = compute_gae_ref(reward, done, value, next_value.clone())
    adv_new, ret_new = compute_gae(reward, done, value, next_value.clone())

    adv_diff = (adv_ref - adv_new).abs().max().item()
    ret_diff = (ret_ref - ret_new).abs().max().item()
    ok = torch.allclose(adv_ref, adv_new, rtol=1e-4, atol=1e-4)
    print(f"  ({N},{T},{k}): adv_diff={adv_diff:.2e} ret_diff={ret_diff:.2e} [{'PASS' if ok else 'FAIL'}]")

print("\n=== Testing compute_gae_ (T, N, k) ===")
for T, N, k in [(32, 16, 1), (64, 64, 4), (128, 256, 1)]:
    reward = torch.randn(T, N, k, device=device)
    done = (torch.rand(T, N, 1, device=device) > 0.95).float()
    value = torch.randn(T, N, k, device=device)
    next_value = torch.randn(N, k, device=device)

    adv_ref, ret_ref = compute_gae_ref_(reward, done, value, next_value.clone())
    adv_new, ret_new = compute_gae_(reward, done, value, next_value.clone())

    adv_diff = (adv_ref - adv_new).abs().max().item()
    ret_diff = (ret_ref - ret_new).abs().max().item()
    ok = torch.allclose(adv_ref, adv_new, rtol=1e-4, atol=1e-4)
    print(f"  ({T},{N},{k}): adv_diff={adv_diff:.2e} ret_diff={ret_diff:.2e} [{'PASS' if ok else 'FAIL'}]")

# Performance
print("\n=== Performance ===")
def bench(fn, warmup=10, runs=50, **kw):
    for _ in range(warmup): fn(**kw)
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(**kw); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return np.mean(times)

for N, T, k in [(64, 64, 4), (256, 128, 1)]:
    reward = torch.randn(N, T, k, device=device)
    done = torch.zeros(N, T, 1, device=device)
    value = torch.randn(N, T, k, device=device)
    nv = torch.randn(N, k, device=device)

    orig = bench(compute_gae_ref, reward=reward, done=done, value=value, next_value=nv.clone())
    triton_t = bench(compute_gae, reward=reward, done=done, value=value, next_value=nv.clone())
    print(f"  compute_gae ({N},{T},{k}): orig={orig:.3f}ms triton={triton_t:.3f}ms speedup={orig/triton_t:.1f}x")
