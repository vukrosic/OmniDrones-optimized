# OmniDrones Triton Kernel Improvement Plan

Generated: 2026-03-28. RTX 3090 24GB, PyTorch 2.11+cu126.

## Current State

All measurements 100-run CUDA Events, median-based:

| Kernel | Size | Speedup | Status |
|--------|------|---------|--------|
| GAE (PPO common.py) | 256x256x1 | 248x | Integrated |
| GAE (utils NTK) | 256x128x1 | 153x | Integrated |
| GAE (utils TNK) | 128x256x1 | 139x | Integrated |
| Reward (fused) | 4096 envs | 4.3x | Standalone only |
| Obs norm | 1024x128 | 4.7x | Standalone only |

## Tier 1: Integrate Standalone Kernels (High Value)

### Task 1.1: Integrate reward kernel into environment task files
- **Target**: `omni_drones/envs/single/hover.py` and similar task files
- **What**: Replace multi-op PyTorch reward computations with `reward_triton()`
- **Expected gain**: 4.3x on reward step, currently unmeasured contribution to training loop
- **How**: Find `reward` computation in env step, replace with Triton call, add fallback
- **Status**: TODO

### Task 1.2: Integrate obs norm kernel into ObservationNormalization module
- **Target**: `omni_drones/learning/` — find RunningMeanStd usage
- **What**: Replace the running mean/std update+normalize with `RunningMeanStdTriton`
- **Expected gain**: 4.7x on obs normalization
- **Status**: TODO

## Tier 2: New Fusion Opportunities

### Task 2.1: Profile full PPO training step
- **Why**: GAE was the bottleneck. Now it's 100x faster. What's next?
- **How**: Add torch.profiler to the PPO `update()` method, measure wall-clock breakdown
- **Expected output**: Ranked list of remaining bottlenecks
- **Status**: TODO

### Task 2.2: Fuse advantage normalization with GAE
- **Target**: After GAE, PPO calls `(adv - adv.mean()) / (adv.std() + eps)` — 3 separate kernels
- **What**: Add a normalize pass into the GAE kernel output, or a fused normalize Triton kernel
- **Expected gain**: 1.5-2x on the normalize step (minor absolute saving)
- **Status**: TODO

### Task 2.3: Fuse value loss computation
- **Target**: Clipped value loss: `(v_pred - returns).pow(2)` with clipping — 4+ ops
- **What**: Single Triton kernel for clipped value loss
- **Expected gain**: 2-3x on value loss (minor absolute)
- **Status**: TODO

## Tier 3: Exploratory

### Task 3.1: torch.compile on full training step
- Now that GAE is fast, compile the remaining forward/backward graph
- Try `torch.compile(mode="reduce-overhead")` on the PPO policy network forward pass
- Expected: 0-1.3x (memory-bound at these sizes, but worth measuring)
- **Status**: TODO

## Not Pursuing

- MLP fwd/bwd: memory-bound, tested, no speedup
- Rotor dynamics: <0.12ms (kernel launch overhead dominates)
- Lee controller: <0.14ms (same)
- Custom matmul: cuBLAS already optimal for these shapes

## Execution Order

1. Task 2.1 (profile) — find out what's actually slow now
2. Task 1.1 (reward integration) — concrete 4.3x win
3. Task 1.2 (obs norm integration) — concrete 4.7x win
4. Tasks 2.2, 2.3 depending on profile results
