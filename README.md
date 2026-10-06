<div align="center">

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>OCBench</h1>
    </summary>
  </ul>
</div>

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-598BE7?style=for-the-badge&logo=python&logoColor=598BE7&labelColor=F0F0F0"/></a> &emsp;
<a href="https://pypi.org/project/ocbench/"><img src="https://img.shields.io/pypi/v/ocbench?style=for-the-badge&labelColor=F0F0F0&color=598BE7"/></a> &emsp;
<a href="https://docs.astral.sh/ruff/"><img src="https://img.shields.io/badge/Code style-ruff-598BE7?style=for-the-badge&labelColor=F0F0F0&color=598BE7"/></a> &emsp;
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-598BE7?style=for-the-badge&labelColor=F0F0F0&color=598BE7"/></a>

![image](assets/env_teaser.png)

<div id="toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h2><a href="https://seohong.me/projects/ocbench/ocbench.pdf">Paper</a> &emsp; <a href="https://seohong.me/projects/ocbench/">Project page</a> &emsp; <a href="https://seohong.me/blog/behavioral-cloning-mystery/">Blog post</a></h2>
    </summary>
  </ul>
</div>

</div>


# Overview

OCBench is a controllable robotic manipulation benchmark for studying behavioral cloning (BC) and reinforcement learning (RL).
OCBench provides GPU-accelerated tasks and scripted policies that mimic the properties of human demonstrations.
This enables controlled studies of BC and RL algorithms with virtually unlimited data.


# Quick start

### Installation

OCBench can be easily installed via PyPI:

```shell
pip install "ocbench[mjwarp,train]"
```

The additional `mjwarp` and `train` dependencies are required for the GPU-accelerated (MJWarp) environments and reference implementations, respectively.
If you only want to use the CPU environments, you can install OCBench without the optional dependencies:

```shell
pip install ocbench
```

### Usage

OCBench environments follow the [Gymnasium](https://gymnasium.farama.org/) interface.
Here's an example of instantiating and running a CPU environment:

```python
import ocbench

# Make an environment.
env = ocbench.make('block-cpu-single-task1-v0')
ob, info = env.reset()

# Run an episode.
ret = 0.0
done = False
while not done:
    action = env.action_space.sample()  # Replace this with your agent's action.
    ob, reward, terminated, truncated, info = env.step(action)
    ret += reward
    done = terminated or truncated

print('Return:', ret)
env.close()
```

OCBench defaults to GPU-accelerated (MJWarp) environments.
Here's an example of running 1024 parallel environments:


```python
import numpy as np
import ocbench

# Make 1024 parallel environments.
env_name = 'block-single-task1-v0'
env = ocbench.make(env_name, nworld=1024)
_, _, _, max_episode_steps = ocbench.parse_env_spec(env_name)
ob, info = env.reset()

# Run 1024 episodes in parallel.
episode_returns = np.zeros(env.nworld)
done = np.zeros(env.nworld, dtype=bool)
for step in range(max_episode_steps):
    action = env.action_space.sample()  # Shape: (1024, 7). Replace with your agent's actions.
    action[done] = 0.0
    ob, reward, terminated, truncated, info = env.step(action)
    episode_returns[~done] += reward[~done]
    done |= terminated | truncated
    if done.all():
        break

print('Mean return:', episode_returns.mean())
env.close()
```


### Cartesian actions with analytic inverse kinematics

Optional wrappers expose a five-dimensional action space
`[dx, dy, dz, dyaw, gripper]`, normalized to `[-1, 1]`:

```python
import numpy as np
import ocbench
from ocbench.wrappers import CartesianActionWrapper, MjWarpCartesianActionWrapper

# Single CPU environment.
env = CartesianActionWrapper(ocbench.make('block-cpu-single-task1-v0'))
ob, info = env.reset(seed=0)
ob, reward, terminated, truncated, info = env.step(np.zeros(5))
env.close()

# Batched MJWarp environment (requires CUDA and the mjwarp extra).
env = MjWarpCartesianActionWrapper(ocbench.make('block-single-task1-v0', nworld=1024))
ob, info = env.reset(seed=0)
ob, reward, terminated, truncated, info = env.step(np.zeros((1024, 5)))
env.close()
```

Translation and yaw are relative to the measured gripper pinch pose, in world
coordinates, with the gripper pointing down. The default per-step scales are
0.05 m for each translation axis, 0.3 rad for yaw, and 0.12 for the gripper;
`action_delta_scale` scales these values (the `lite` variants use a factor of 5).
Positive gripper actions close the fingers. Position targets are clipped to the
environment's workspace bounds, and yaw wraps naturally through rotations.

The solver evaluates up to eight analytic branches for the bundled UR5e model,
checks their forward kinematics, and selects the nearest solution within actuator
limits. The wrapper converts the pinch target to the arm's attachment-site frame
and maps the solution through the existing joint-action limits. Consequently,
a Cartesian target may require multiple steps to reach. If no valid solution
exists, the arm holds its current joint positions while the gripper command
remains active; `info['ik_no_solution']` reports this per environment.
The solver uses the bundled model dimensions and does not check collisions.

The CPU implementation uses NumPy; batched IK uses native Warp kernels. Neither
requires JAX. The MJWarp wrapper also provides `action_gpu(actions)` and
`step_cartesian_actions_gpu(actions, done=None)` for `wp.float32` arrays of shape
`(nworld, 5)` on the simulation device. These avoid copying IK actions to the
host; the normal `step` interface retains OCBench's NumPy observations and
transfers. `ik_no_solution_gpu` exposes the device failure flags. Device output
buffers are reused on the next call. CPU/NumPy actions reject nonfinite inputs;
nonfinite device action rows hold the arm and gripper and set the failure flag.

Wrappers preserve observations, rewards, and episode handling. Unwrapped
environments, reset IK, and scripted policies retain their existing behavior.
Released datasets use seven-dimensional joint actions and are not directly
compatible with the Cartesian action space.

For standalone attachment-site IK, use
`ocbench.controllers.AnalyticIKController().solve_with_status(pos, quat, curr_qpos)`.
The quaternion convention is `(w, x, y, z)`; the result is `(qpos_target, no_solution)`.
Optional `joint_limits` passed to the controller have shape `(6, 2)`.

To run the kinematics and wrapper tests:

```shell
pip install -e ".[dev]"
MUJOCO_GL=disable python -m pytest tests
```

Installing the `mjwarp` extra also enables Warp parity tests on CPU. CUDA parity
and simulation tests run when a CUDA device is available, otherwise they skip.


# Tasks and datasets

### Tasks

OCBench provides 28 manipulation tasks across 5 types of environments.
See the [paper](https://seohong.me/projects/ocbench/ocbench.pdf) for more details.

| Environment | Tasks |
| --- | --- |
| `block-single` | `task1` |
| `block-double` | `task1`, `task2` |
| `block-triple` | `task1`, `task2` |
| `block-quadruple` | `task1`, `task2`, `task3` |
| `chamber-easy` | `task1`, `task2`, `task3` |
| `chamber-medium` | `task1`, `task2` |
| `chamber-hard` | `task1`, `task2` |
| `switch-3x3` | `task1`, `task2` |
| `switch-4x4` | `task1`, `task2` |
| `switch-5x5` | `task1`, `task2` |
| `hanoi-single` | `task1`, `task2` |
| `hanoi-double` | `task1`, `task2` |
| `hanoi-triple` | `task1`, `task2` |
| `bowling` | `task1` |

OCBench also provides several variants of the environments:

* `lite`: The `lite` variant has much shorter horizons and simplified scripted policies. This can be useful for faster prototyping.
* `cpu`: The `cpu` variant uses CPU-based MuJoCo instead of GPU-based MJWarp.
* `visual`: The `visual` variant provides pixel-based observations instead of state-based observations.

| Environment name | `lite` | `cpu` | `visual` |
| --- | --- | --- | --- |
| `block-single-task1-v0` | No | No | No |
| `block-lite-single-task1-v0` | Yes | No | No |
| `block-cpu-single-task1-v0` | No | Yes | No |
| `block-cpu-lite-single-task1-v0` | Yes | Yes | No |
| `visual-block-single-task1-v0` | No | No | Yes |
| `visual-block-lite-single-task1-v0` | Yes | No | Yes |
| `visual-block-cpu-single-task1-v0` | No | Yes | Yes |
| `visual-block-cpu-lite-single-task1-v0` | Yes | Yes | Yes |

### Datasets

OCBench datasets are hosted on [Hugging Face](https://huggingface.co/datasets/seohongpark/ocbench).
The datasets will be automatically downloaded to `~/.cache/ocbench/datasets` by default.
You can also specify a dataset root directory or path by setting `dataset_path` or `dataset_root`.

For `lite`, `cpu`, and `visual` variants, we only provide datasets for selected tasks.
In particular, we do not provide datasets for full-horizon `cpu` environments.
However, you can still generate datasets for these variants using the provided data-generation script.

Here's an example of how to load an environment and its datasets:

```python
import ocbench

# Make an environment and load datasets.
env_name = 'block-single-task1-v0'
env, train_dataset, val_dataset = ocbench.make_env_and_datasets(
    env_name,  # Environment name.
    dataset_root='~/.cache/ocbench/datasets',  # Directory to save datasets (optional).
    nworld=1024,  # Number of parallel environments (optional).
)

# Example: Episode 1 has 3 steps and terminates early;
# episode 2 has 4 steps and reaches the time limit.
#
#                  |<-episode 1->|  |<---episode 2--->|
# ---------------------------------------------------------------
# 'observations': [s0, s1, s2, s3,  s0, s1, s2, s3, s4,  ...]
# 'actions'     : [a0, a1, a2,      a0, a1, a2, a3,      ...]
# 'rewards'     : [r0, r1, r2,      r0, r1, r2, r3,      ...]
# 'terminals'   : [ 0,  0,  1,       0,  0,  0,  1,      ...]
# 'masks'       : [ 1,  1,  0,       1,  1,  1,  1,      ...]
#
# Each episode includes its final observation, but no extra action, so
# the 'observations' and 'actions' arrays can have different lengths.
# For observation_interval > 1 (i.e., sparse datasets), observations are stored
# at steps 0, interval, 2 * interval, ..., plus the final observation.
```

We provide scripted policies and a data-generation script in
[`data_gen_scripts/generate_ocbench.py`](data_gen_scripts/generate_ocbench.py).
We provide the exact commands for the datasets used in the paper in [commands.sh](data_gen_scripts/commands.sh).

### Streaming data

OCBench also supports streaming (on-the-fly) data generation for MJWarp environments to train an agent with virtually unlimited data.
Set `--streaming_data=1` and related flags in `main.py` to use this feature.


# Reference implementations

We provide reference implementations of several BC and RL algorithms in the `impls` directory.

### Installation

After cloning this repository, run the following from its root directory:

```shell
cd impls
pip install "ocbench[mjwarp,train]"
```

By default, this uses the PyPI version of OCBench.
To use a local version of OCBench (e.g., for training on modified environments), run the following instead from the repository root:

```shell
pip install -e ".[mjwarp,train]"
```

### Algorithms

- **BC** ([`bc.py`](impls/agents/bc.py)): Deterministic behavioral cloning.
- **FBC** ([`fbc.py`](impls/agents/fbc.py)): Flow behavioral cloning.
- **FSQBC** ([`fsqbc.py`](impls/agents/fsqbc.py)): FSQ behavioral cloning.
- **History-conditioned FBC** ([`history_fbc.py`](impls/agents/history_fbc.py)): History-conditioned flow behavioral cloning.
- **IFQL** ([`ifql.py`](impls/agents/ifql.py)): Implicit flow Q-learning (the flow variant of implicit diffusion Q-learning).
- **FQL** ([`fql.py`](impls/agents/fql.py)): Flow Q-learning.
- **DSRL** ([`dsrl.py`](impls/agents/dsrl.py)): Diffusion steering via reinforcement learning (DSRL-NA).
- **Advantage conditioning** ([`binary_advc.py`](impls/agents/binary_advc.py)): Binary advantage-conditioned flow behavioral cloning.

We provide commands to reproduce the benchmark tables in the paper in [commands.sh](impls/commands.sh).


# Acknowledgments

This codebase is inspired by or partly uses code from the following repositories:

- [OGBench](https://github.com/seohongpark/ogbench) for the general infrastructure.
- [MuJoCo](https://github.com/google-deepmind/mujoco) and [MJWarp](https://github.com/google-deepmind/mujoco_warp) for simulation.
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) for the robot descriptions (Universal Robots UR5e and Robotiq 2F-85).
- [ur-analytic-ik](https://github.com/Victorlouisdg/ur-analytic-ik) for the analytic UR kinematics equations (MIT; see [license](ocbench/controllers/ur_analytic_ik.LICENSE)).
- [Meta-World](https://github.com/Farama-Foundation/Metaworld) for the objects (drawer, window, and button) in the manipulation environments.


# Citation

```bibtex
@misc{ocbench_park2026,
  title={{Behavioral Cloning Mystery}},
  author={Park, Seohong and Levine, Sergey},
  year={2026},
  url={https://seohong.me/projects/ocbench/ocbench.pdf},
}
```
