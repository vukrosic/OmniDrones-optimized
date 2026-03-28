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
# 4. New Triton Kernels: quaternion ops, expln, hover reward
# ============================================================
print("\n=== New Triton Kernels ===")

# --- quat_rotate / quat_rotate_inverse ---
import functools

def _manual_batch(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        batch_shapes = set(arg.shape[:-1] for arg in args if isinstance(arg, torch.Tensor))
        batch_shape = batch_shapes.pop()
        args = tuple(arg.reshape(-1, arg.shape[-1]) if isinstance(arg, torch.Tensor) else arg for arg in args)
        out = func(*args, **kwargs)
        return out.unflatten(0, batch_shape)
    return wrapped

@_manual_batch
def quat_rotate_ref(q, v):
    shape = q.shape
    q_w = q[:, 0]; q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

@_manual_batch
def quat_rotate_inv_ref(q, v):
    shape = q.shape
    q_w = q[:, 0]; q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

from omni_drones.utils.torch import quat_rotate, quat_rotate_inverse, quat_mul
from omni_drones.utils.torch import quaternion_to_rotation_matrix

for N in [64, 256, 1024]:
    q = torch.randn(N, 4, device=device); q = q / q.norm(dim=-1, keepdim=True)
    v = torch.randn(N, 3, device=device)
    ref_r = quat_rotate_ref(q, v)
    out_r = quat_rotate(q, v)
    ref_i = quat_rotate_inv_ref(q, v)
    out_i = quat_rotate_inverse(q, v)
    check(f"quat_rotate(N={N})", ref_r, out_r)
    check(f"quat_rotate_inverse(N={N})", ref_i, out_i)

for N in [64, 256, 1024]:
    q = torch.randn(N, 4, device=device); q = q / q.norm(dim=-1, keepdim=True)
    def quat_to_rotmat_ref(quaternion):
        w, x, y, z = torch.unbind(quaternion, dim=-1)
        tx = 2.0*x; ty = 2.0*y; tz = 2.0*z
        twx=tx*w; twy=ty*w; twz=tz*w; txx=tx*x; txy=ty*x; txz=tz*x; tyy=ty*y; tyz=tz*y; tzz=tz*z
        m = torch.stack([1-(tyy+tzz), txy-twz, txz+twy, txy+twz, 1-(txx+tzz), tyz-twx,
                         txz-twy, tyz+twx, 1-(txx+tyy)], dim=-1)
        return m.unflatten(m.dim()-1, (3,3))
    ref = quat_to_rotmat_ref(q)
    out = quaternion_to_rotation_matrix(q)
    check(f"quat_to_rotation_matrix(N={N})", ref, out)

for N in [64, 256, 1024]:
    def quat_mul_ref(a, b):
        shape = a.shape; a = a.reshape(-1, 4); b = b.reshape(-1, 4)
        w1,x1,y1,z1 = a[:,0],a[:,1],a[:,2],a[:,3]; w2,x2,y2,z2 = b[:,0],b[:,1],b[:,2],b[:,3]
        ww=(z1+x1)*(x2+y2); yy=(w1-y1)*(w2+z2); zz=(w1+y1)*(w2-z2); xx=ww+yy+zz
        qq=0.5*(xx+(z1-x1)*(x2-y2))
        w=qq-ww+(z1-y1)*(y2-z2); x=qq-xx+(x1+w1)*(x2+w2); y=qq-yy+(w1-x1)*(y2+z2); z=qq-zz+(z1+y1)*(w2-x2)
        return torch.stack([w,x,y,z], dim=-1).reshape(shape)
    a = torch.randn(N, 4, device=device); a = a / a.norm(dim=-1, keepdim=True)
    b = torch.randn(N, 4, device=device); b = b / b.norm(dim=-1, keepdim=True)
    ref = quat_mul_ref(a, b); out = quat_mul(a, b)
    check(f"quat_mul(N={N})", ref, out)

# --- expln ---
from omni_drones.learning.modules.distributions import expln

def expln_ref(x):
    out = torch.empty_like(x)
    idx_neg = x <= 0
    out[idx_neg] = x[idx_neg].exp()
    out[~idx_neg] = x[~idx_neg].log1p() + 1
    return out

for N in [1024, 32768]:
    x = torch.randn(N, device=device)
    check(f"expln(N={N})", expln_ref(x), expln(x))

# --- hover_reward_triton ---
from triton_kernels import hover_reward_triton

def hover_reward_ref(rpos, rheading, up_z, spinnage, effort, throttle_diff,
                     ds=1.2, ew=0.1, sw=0.1):
    distance = torch.norm(torch.cat([rpos, rheading], dim=-1), dim=-1)
    rp = 1.0 / (1.0 + (ds * distance) ** 2)
    ru = ((up_z + 1) / 2) ** 2
    rs = 1.0 / (1.0 + spinnage ** 2)
    re = ew * torch.exp(-effort)
    rsm = sw * torch.exp(-throttle_diff)
    return rp + rp * (ru + rs) + re + rsm

for N in [256, 1024, 4096]:
    rpos = torch.randn(N, 3, device=device)
    rheading = torch.randn(N, 3, device=device)
    up_z = torch.randn(N, device=device)
    spinnage = torch.rand(N, device=device).abs()
    effort = torch.rand(N, device=device).abs()
    throttle_diff = torch.rand(N, device=device).abs()
    ref = hover_reward_ref(rpos, rheading, up_z, spinnage, effort, throttle_diff)
    out = hover_reward_triton(rpos, rheading, up_z, spinnage, effort, throttle_diff)
    check(f"hover_reward_triton(N={N})", ref, out)

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
