# VAIC Assets

This directory contains the simulation assets for [VAIC](https://vaic-humanoid.github.io/): the G1 humanoid robot, the interactive objects used in our experiments, and the configuration code that loads them. All assets are authored for [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) / [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and are self-contained USD files (no external references).

## Layout

```
assets/
├── __init__.py              # ROBOTS / OBJECTS registries + asset metadata helper
├── base.py                  # shared ArticulationCfg utilities
├── spawn.py                 # custom spawners / cloning helpers
├── g1.py                    # G1 robot ArticulationCfg
├── objects.py               # object Rigid/Articulation configs
├── default_environment.usd  # default ground / lighting scene
├── g1/                       # G1 robot USDs
└── objects/                  # interactive object USDs
```

## Robot

The G1 humanoid is provided in four collision/end-effector variants under `g1/`. The variant is selected at runtime via the `ROBOT_TYPE` placeholder in `g1.py`.

| Variant | File |
| --- | --- |
| 29-DoF, rubber hand, box feet, box eef, capsule body | `g1/g1_29dof_rubberhand-feet_box-eef_box-body_capsule.usd` |
| 29-DoF, rubber hand, sphere feet, box eef, capsule body | `g1/g1_29dof_rubberhand-feet_sphere-eef_box-body_capsule.usd` |

## Objects

| Object | Path | Type | Config |
| --- | --- | --- | --- |
| Box | `objects/mybox/mybox.usd` | Rigid body | `MYBOX_CFG` |
| Skateboard | `objects/skateboard/skateboard.usd` | Articulated (4 wheels) | `SKATEBOARD_CFG` |
| Cart | `objects/cart/cart.usd` | Articulated (8 wheels) | `CART_CFG` |
| Wheelchair | `objects/wheelchair/wheelchair.usd` | Articulated (casters) | `WHEELCHAIR_CFG` |
| Stair slope | `objects/stairslope/stairslope.usd` | Static (kinematic) | `STAIRSLOPE_CFG` |

## Usage

The asset configs are registered in `__init__.py`:

```python
from active_adaptation.assets import ROBOTS, OBJECTS

robot_cfg = ROBOTS["g1"]          # G1 ArticulationCfg
object_cfg = OBJECTS["skateboard"] # object config by name
```

`ROBOTS["g1"]` and the wheeled objects use `{ROBOT_TYPE}` / `{OBJECT_TYPE}` placeholders in their `usd_path`; these are substituted with the concrete variant name at scene-build time.
