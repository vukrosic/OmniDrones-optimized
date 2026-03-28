"""
Comprehensive correctness test suite for OmniDrones Triton kernels.
Run sparingly — only after kernel changes, not during iteration.

Usage: gpu py test_omnidrones_kernels.py
"""
import sys
import torch
import numpy as np

sys.modules["isaacsim"] = type(sys)("isaacsim")
sys.modules["isaacsim"].SimulationApp = None
sys.path.insert(0, "/workspace/OmniDrones")

device = torch.device("cuda")
torch.manual_seed(42)

PASS = 0
FAIL = 0

def check(name, ref, out, rtol=1e-4, atol=1e-4):
    global PASS, FAIL
    diff = (ref - out).abs().max().item()
    ok = torch.allclose(ref, out, rtol=rtol, atol=atol)
    if ok:
        PASS += 1
        print(f"  PASS {name} (max_diff={diff:.2e})")
    else:
        FAIL += 1
        print(f"  FAIL {name} (max_diff={diff:.2e})")

# ============================================================
# 1. PPO GAE (common.py) — (batch, steps, agents, 1) layout
# ============================================================
print("=== PPO GAE (common.py) ===")
from omni_drones.learning.ppo.common import GAE

# Reference Python implementation
def gae_ref(reward, terminated, value, next_value, gamma=0.99, lmbda=0.95):
    num_steps = terminated.shape[1]
    advantages = torch.zeros_like(reward)
    not_done = 1 - terminated.float()
    gae = 0
    for step in reversed(range(num_steps)):
        delta = reward[:, step] + gamma * next_value[:, step] * not_done[:, step] - value[:, step]
        advantages[:, step] = gae = delta + (gamma * lmbda * not_done[:, step] * gae)
    returns = advantages + value
    return advantages, returns

gae_mod = GAE(0.99, 0.95).to(device)
for batch, steps, agents in [(4, 32, 1), (16, 32, 4), (64, 64, 4), (256, 128, 1)]:
    r = torch.randn(batch, steps, agents, 1, device=device)
    t = (torch.rand(batch, steps, agents, 1, device=device) > 0.95).float()
    v = torch.randn(batch, steps, agents, 1, device=device)
    nv = torch.randn(batch, steps, agents, 1, device=device)
    adv_ref, ret_ref = gae_ref(r, t, v, nv)
    adv_new, ret_new = gae_mod(r, t, v, nv)
    check(f"ppo_gae({batch},{steps},{agents},1)_adv", adv_ref, adv_new)
    check(f"ppo_gae({batch},{steps},{agents},1)_ret", ret_ref, ret_new)

# ============================================================
# 2. Utils GAE — compute_gae (N, T, k) and compute_gae_ (T, N, k)
# ============================================================
print("\n=== Utils GAE (compute_gae, compute_gae_) ===")
from omni_drones.learning.utils.gae import compute_gae, compute_gae_

def compute_gae_ref(reward, done, value, next_value, gamma=0.99, lmbda=0.95):
    not_done = 1.0 - done.float()
    _, num_steps = not_done.shape[:2]
    gae = 0
    advantages = torch.zeros_like(reward)
    for step in reversed(range(num_steps)):
        delta = reward[:, step] + gamma * next_value * not_done[:, step] - value[:, step]
        advantages[:, step] = gae = delta + (gamma * lmbda * not_done[:, step] * gae)
        next_value = value[:, step]
    return advantages, advantages + value

for N, T, k in [(16, 32, 1), (64, 64, 4), (256, 128, 1)]:
    r = torch.randn(N, T, k, device=device)
    d = (torch.rand(N, T, 1, device=device) > 0.95).float()
    v = torch.randn(N, T, k, device=device)
    nv = torch.randn(N, k, device=device)
    adv_ref, ret_ref = compute_gae_ref(r, d, v, nv.clone())
    adv_new, ret_new = compute_gae(r, d, v, nv.clone())
    check(f"compute_gae({N},{T},{k})_adv", adv_ref, adv_new)

def compute_gae_ref_(reward, done, value, next_value, gamma=0.99, lmbda=0.95):
    not_done = 1.0 - done.float()
    num_steps = not_done.shape[0]
    gae = 0
    advantages = torch.zeros_like(reward)
    for step in reversed(range(num_steps)):
        delta = reward[step] + gamma * next_value * not_done[step] - value[step]
        advantages[step] = gae = delta + (gamma * lmbda * not_done[step] * gae)
        next_value = value[step]
    return advantages, advantages + value

for T, N, k in [(32, 16, 1), (64, 64, 4), (128, 256, 1)]:
    r = torch.randn(T, N, k, device=device)
    d = (torch.rand(T, N, 1, device=device) > 0.95).float()
    v = torch.randn(T, N, k, device=device)
    nv = torch.randn(N, k, device=device)
    adv_ref, _ = compute_gae_ref_(r, d, v, nv.clone())
    adv_new, _ = compute_gae_(r, d, v, nv.clone())
    check(f"compute_gae_({T},{N},{k})_adv", adv_ref, adv_new)

# ============================================================
# 3. Triton kernels standalone (triton_kernels.py)
# ============================================================
print("\n=== Standalone Triton Kernels ===")
sys.path.insert(0, "/workspace/OmniDrones")
from triton_kernels import gae_triton, RunningMeanStdTriton, reward_triton

# GAE standalone
for T, N in [(32, 64), (128, 1024)]:
    r = torch.randn(T, N, device=device)
    v = torch.randn(T, N, device=device)
    d = (torch.rand(T, N, device=device) > 0.95).float()
    # Reference
    adv_ref = torch.zeros_like(r)
    last_gae = torch.zeros(N, device=device)
    for t in reversed(range(T)):
        nv = v[t+1] if t < T-1 else torch.zeros(N, device=device)
        nd = 1.0 - d[t]
        delta = r[t] + 0.99 * nv * nd - v[t]
        last_gae = delta + 0.99 * 0.95 * nd * last_gae
        adv_ref[t] = last_gae
    adv_new = gae_triton(r, v, d)
    check(f"gae_triton({T},{N})", adv_ref, adv_new)

# Reward
for N in [256, 4096]:
    pos = torch.randn(N, 3, device=device)
    tp = torch.randn(N, 3, device=device)
    vel = torch.randn(N, 3, device=device)
    av = torch.randn(N, 3, device=device)
    eff = torch.randn(N, 4, device=device)
    ref = 1.0/(1.0 + (pos-tp).norm(dim=-1)**2) - 0.1*vel.norm(dim=-1) - 0.05*av.norm(dim=-1) - 0.01*eff.norm(dim=-1)
    out = reward_triton(pos, tp, vel, av, eff)
    check(f"reward_triton(N={N})", ref, out)

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*40}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
    sys.exit(1)
