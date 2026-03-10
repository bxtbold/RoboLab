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
│       ├── isaac/
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

```python
from robolab_backend.isaac import parse_args, IsaacSession

parser.add_argument("--task", default="Robolab-Isaac-FrankaPush-Image-v0")
args = parse_args()   # must be before any isaaclab imports

with IsaacSession(args) as session:
    import robolab_envs.isaac.manipulation.franka_push  # registers the gym task
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
```

## Current Tasks

| ID | Obs | Action | Reward |
|---|---|---|---|
| `Robolab-Isaac-FrankaPush-Image-v0` | proprioception (6D) + table_cam + wrist_cam (64×64) | 3-DoF IK | sparse +1 |

## Run

```bash
# quick test — steps env and saves table_cam video
python scripts/test.py --headless --num_envs 2 --num_steps 200
```

---

## TODO

### Higher prioprities
- [ ] Integration with runners
- [ ] Franka push state variant (`Robolab-Isaac-FrankaPush-State-v0`)
- [ ] Real robot backend (`RealRobotSession` - franka)
- [ ] RecordingWrapper

### Mid prioprities
- [ ] Add teleop devices (keyboard and space mouse)
- [ ] Achieve a better package management
- [ ] Data collection script (teleop -> `.pt` demos)
- [ ] Franka pick task (image + state)
- [ ] `get_env()` factory in `robolab_envs`
- [ ] MuJoCo backend (`MujocoSession`)
- [ ] Add external benchmarks ([furniture-bench](https://clvrai.github.io/furniture-bench/), [metaworld](https://metaworld.farama.org/))
- [ ] Domain randomization

### Lower prioprities
- [ ] Versioning (OS & packages etc)
- [ ] Update docs/guides on
    - [ ] README.md
    - [ ] installing isaaclab
    - [ ] integrating benchmarks
    - [ ] adding a custom robot
