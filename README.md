# RoboLab

Modular robotics framework for Isaac, MuJoCo, and real robots. Clean separation between environment definitions, simulator backends, and learners.

---

## Packages

| Package | Purpose |
|---|---|
| `robolab_envs` | Task/environment definitions, gym registration, MDP logic |
| `robolab_backend` | Simulator & hardware lifecycle (Isaac, MuJoCo, real) |

> TODO: update packages after implementing other packages

## Structure

```
source/
├── robolab_envs/
│   └── robolab_envs/
│       ├── isaac/manipulation/franka_push/
│       ├── isaac/manipulation/franka_pick/
│       ├── mujoco/
│       └── real/
├── robolab_backend/
│   └── robolab_backend/
│       ├── isaac/          # parse_args, IsaacSession, AppLauncher
│       ├── mujoco/
│       └── real/
```
> TODO: update structure after implementing other packages

## Install

```bash
pip install -e source/robolab_envs
pip install -e source/robolab_backend
```
> TODO: make it easier to install

## Usage

### Basic env setup

```python
import gymnasium as gym
import argparse
from robolab_backend.isaac import parse_args, IsaacSession

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-FrankaPush-Image-v0")
parser.add_argument("--num-envs", type=int, default=1)
args = parse_args(parser)   # must be before any isaaclab imports

with IsaacSession(args) as session:
    import robolab_envs.isaac.manipulation.franka_push  # registers the gym task
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
```

## Current Tasks

| ID | Obs | Action | Reward |
|---|---|---|---|
| `Isaac-FrankaPush-Image-v0` | proprioception (6) + table_cam + wrist_cam (64×64x3) | 3-DoF IK delta | sparse +1 on success |

---

## TODO

### High priority
- [ ] Integration with runners
- [ ] Franka push state variant (`Isaac-FrankaPush-State-v0`)
- [ ] Real robot backend (`RealRobotSession` - franka)
- [ ] RecordingWrapper

### Mid priority
- [ ] Teleop devices (keyboard, SpaceMouse)
- [ ] Data collection script (teleop → `.pt` demos)
- [ ] Franka pick task (image + state)
- [ ] MuJoCo backend (`MujocoSession`)
- [ ] `get_env()` factory in `robolab_envs`
- [ ] External benchmarks ([furniture-bench](https://clvrai.github.io/furniture-bench/), [metaworld](https://metaworld.farama.org/))
- [ ] Domain randomization

### Low priority
- [ ] `pyproject.toml`-based installs (replace legacy `setup.py`)
- [ ] Versioning (OS & package pins)
- [ ] Guides: installing IsaacLab, integrating benchmarks, adding a custom robot
