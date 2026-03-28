![Visualization of OmniDrones](docs/source/_static/visualization.jpg)

# OmniDrones-Optimized

This is a sped-up version of [btx0424/OmniDrones](https://github.com/btx0424/OmniDrones), focused on reducing GPU overhead in the PPO/MAPPO update path while preserving training semantics.

Current verified headline result on the synthetic PPO update benchmark (`B=64`, `T=128`, `N=4`, `D_obs=64`, `D_act=4`, RTX 3090):

- **2.63x faster** end-to-end than baseline without TF32
- **2.73x faster** end-to-end than baseline with TF32 enabled

What was changed in this fork:

- Triton CUDA kernels for PPO GAE and utility GAE paths
- Fused standalone Triton kernels for reward and normalization-heavy paths
- Optional TF32 support for PPO/MAPPO MLPs
- Autograd-safe PyTorch fallbacks for gradient-carrying helpers
- TorchRL compatibility fixes for the currently tested stack

Important scope note:

- These headline numbers measure the full PPO **update step**: GAE, advantage normalization, actor/critic forward, loss construction, backward, grad clipping, and optimizer step
- They do **not** include Isaac Sim environment stepping, so simulator-heavy training runs may see smaller overall wall-clock gains

---

## Upstream Project Notes

I greatly appreciate the interest by the community in this project. However, due to several difficulties, this version of the project is hard to maintain and update anymore. I sincerely apologize for the inconvenience. There may or may not be a clearner refactored version in the future. If you believe it is highly helpful to your research, you are welcomed to contact me by emailing to btx0424@outlook.com.

## OmniDrones

[![IsaacSim](https://img.shields.io/badge/Isaac%20Sim-4.1.0-orange.svg)](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.7.html)
[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://omnidrones.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Discord Forum](https://dcbadge.vercel.app/api/server/J4QvXR6tQj)](https://discord.gg/J4QvXR6tQj)

*OmniDrones* is an open-source platform designed for reinforcement learning research on multi-rotor drone systems. Built on [Nvidia Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html), *OmniDrones* features highly efficient and flexible simulation that can be adopted for various research purposes. We also provide a suite of benchmark tasks and algorithm baselines to provide preliminary results for subsequent works.

For usage and more details, please refer to the [documentation](https://omnidrones.readthedocs.io/en/latest/). Unfortunately, it does not support Windows.

Welcome to join our [Discord](https://discord.gg/J4QvXR6tQj) for discussions and questions!

## Notice

The initial release of **OmniDrones** is developed based on Isaac Sim 2022.2.0. It can be found at the [release](https://github.com/btx0424/OmniDrones/tree/release) branch. The current version is developed based on Isaac Sim 4.1.0.

## Announcement 2023-09-25

The initial release of **OmniDrones** is developed based on Isaac Sim 2022.2.0. As the next version of
Isaac Sim (2023.1.0) is expected to bring substantial changes but is not yet available, the APIs and usage
of **OmniDrones** are subject to change. We will try our best to keep the documentation up-to-date.

## Announcement 2023-10-25

The new release of Isaac Sim (2023.1.0) has brought substantial changes as well as new possibilities, among
which the most important is new sensors. We are actively working on it at the `devel` branch. The `release`
branch will still be maintained for compatibility. Feel free to raise issues if you encounter any problems
or have ideas to discuss.


## Optimization Details

The goal of the fork is to reduce GPU overhead in the RL update path without silently changing training semantics. The main work targets PPO/MAPPO bottlenecks, especially GAE, plus a few hot utility kernels used in reward computation and quaternion math.

### What This Fork Changes

- Replaces Python-loop GAE implementations with [Triton](https://github.com/openai/triton) CUDA kernels in:
  - `omni_drones/learning/ppo/common.py`
  - `omni_drones/learning/utils/gae.py`
- Adds fused standalone Triton kernels in `triton_kernels.py` for:
  - GAE
  - reward computation
  - observation normalization
  - hover reward
- Adds optional TF32 support for PPO/MAPPO MLPs via config (`allow_tf32=true`) instead of mutating global backend flags at import time.
- Keeps autograd-safe behavior by falling back to PyTorch for gradient-carrying paths such as `expln` and quaternion helpers.
- Updates TorchRL compatibility for the current tested stack (`torchrl==0.11.1`).

### How The Speedups Were Achieved

#### 1. GAE kernel launch elimination

Upstream GAE is written as a reverse Python loop over time steps. On CUDA that means one launch per step. This fork moves the reverse scan into Triton kernels, parallelizing across environments/agents while keeping the recurrence inside a single kernel launch.

#### 2. Fused elementwise kernels

A few reward and normalization paths are dominated by chains of small tensor ops. This fork fuses those into standalone Triton kernels to reduce launch overhead and intermediate tensor traffic.

#### 3. Optional TF32 for matrix multiplies

For Ampere+ GPUs, actor/critic forward passes can use TF32 matmuls. This is exposed as an explicit PPO/MAPPO config option rather than being forced on import.

#### 4. Correctness-preserving fallbacks

Some utility functions are used in differentiable code paths. Triton kernels that do not provide backward definitions are only used for no-grad CUDA inputs; otherwise the code falls back to standard PyTorch to preserve upstream autograd behavior.

### Benchmarks

Latest benchmark rerun was executed with:

- GPU: RTX 3090 24GB
- PyTorch: `2.11.0+cu126`
- TorchRL: `0.11.1`
- Method: CUDA Events, `100` measured runs, `20` warmup runs
- Command: `python benchmark_rigorous.py`

#### PPO GAE

Median timings from the latest rerun:

| Shape | Python (ms) | Triton (ms) | Speedup | Max Diff |
|------|-------------:|------------:|--------:|---------:|
| `4x32x1` | `3.067` | `0.095` | **32.2x** | `1.07e-06` |
| `16x64x4` | `6.101` | `0.096` | **63.4x** | `2.38e-06` |
| `64x128x4` | `12.037` | `0.096` | **125.1x** | `3.34e-06` |
| `256x256x1` | `23.936` | `0.113` | **211.6x** | `5.72e-06` |

#### Utility GAE

Representative results from `compute_gae` / `compute_gae_`:

| Kernel | Shape | Python (ms) | Triton (ms) | Speedup | Max Diff |
|--------|------|-------------:|------------:|--------:|---------:|
| `compute_gae` | `N256_T128_k1` | `11.756` | `0.078` | **151.1x** | `1.91e-06` |
| `compute_gae` | `N512_T256_k4` | `23.942` | `0.213` | **112.4x** | `3.34e-06` |
| `compute_gae_` | `T128_N256_k1` | `10.845` | `0.078` | **139.3x** | `1.91e-06` |
| `compute_gae_` | `T256_N512_k4` | `24.198` | `0.306` | **79.0x** | `3.34e-06` |

#### Standalone reward kernel

| Envs | Python (ms) | Triton (ms) | Speedup | Max Diff |
|------|-------------:|------------:|--------:|---------:|
| `256` | `0.188` | `0.043` | **4.4x** | `1.49e-07` |
| `1024` | `0.190` | `0.046` | **4.1x** | `1.19e-07` |
| `4096` | `0.188` | `0.043` | **4.4x** | `1.64e-07` |
| `16384` | `0.189` | `0.044` | **4.3x** | `1.79e-07` |

#### Other measured optimizations

Earlier profiling in this fork also measured:

| Component | Baseline (ms) | Optimized (ms) | Speedup | Method |
|-----------|--------------:|---------------:|--------:|--------|
| Actor MLP forward | `1.62` | `0.84` | **2.05x** | TF32 matmul |
| Critic MLP forward | `1.13` | `0.79` | **1.43x** | TF32 matmul |
| Advantage normalization | `0.087` | `0.054` | **1.63x** | Triton fused kernel |

### Important Benchmark Caveat

These are mostly kernel-level or rollout-side wins. They do not translate 1:1 into end-to-end PPO training speedups because backward passes, optimizer steps, and environment stepping still consume a large fraction of wall time.

The strongest win in this fork is GAE. The overall training-step speedup will depend on your rollout length, number of environments, batch size, and how much of the wall clock is spent outside GAE.

For the synthetic PPO update benchmark used during development, the measured full-step speedups were:

| Mode | Mean (ms) | Median (ms) | Speedup vs baseline |
|------|----------:|------------:|--------------------:|
| Baseline | `18.143` | `17.907` | `1.00x` |
| Optimized, TF32 off | `6.823` | `6.818` | **2.63x** |
| Optimized, TF32 on | `6.576` | `6.552` | **2.73x** |

### Correctness And Validation

- `tests/test_omnidrones_kernels.py` now passes with `70/70` checks on the remote benchmark machine.
- Tests cover:
  - forward numerical equivalence
  - numerical-stability edge cases
  - gradient-preserving behavior for autograd-sensitive helpers
  - TorchRL transform compatibility
- Gradient-requiring paths use PyTorch fallbacks when Triton would otherwise drop `grad_fn`.
- CPU and non-Triton environments still use Python/PyTorch fallbacks.

### How To Use The Optimized Paths

- GAE acceleration is automatic on CUDA when Triton is available.
- TF32 is opt-in in PPO/MAPPO config:

```yaml
algo:
  allow_tf32: true
```

- To rerun validation and benchmarks:

```bash
# Correctness / regression checks
python tests/test_omnidrones_kernels.py

# Benchmark suite
python benchmark_rigorous.py
```

Full benchmark outputs are written to `benchmark_results.json`.

## Citation

Please cite [this paper](https://arxiv.org/abs/2309.12825) if you use *OmniDrones* in your work:

```bibtex
@misc{xu2023omnidrones,
    title={OmniDrones: An Efficient and Flexible Platform for Reinforcement Learning in Drone Control},
    author={Botian Xu and Feng Gao and Chao Yu and Ruize Zhang and Yi Wu and Yu Wang},
    year={2023},
    eprint={2309.12825},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```

## Acknowledgement

Some of the abstractions and implementation was heavily inspired by [Isaac Lab](https://github.com/isaac-sim/IsaacLab).
