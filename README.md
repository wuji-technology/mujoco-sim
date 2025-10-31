# WujiHand MuJoCo Simulation and Control

A minimal demo for loading and controlling the WujiHand model in the MuJoCo simulator.

## Requirements

* Python 3.8+ (recommend tested environment)
* Python packages: `pip install mujoco numpy`

## Quick start

### Run the simulation of WujiHand model
Run the simulation GUI for the default (right) hand:
```
python demo_sim.py
```

In the MuJoCo GUI, use the "control" panel (top-right) to manually move joints.

You can use parameters `-s` and `--side` in command line to specify hand.
```bash
python demo_sim.py -s left
# or
python demo_sim.py --side left
```

### Play predefined trajectories
Load and play one of the provided trajectories

```
python demo_trajectory.py -t trajectory/fist.npz
python demo_trajectory.py -t trajectory/pinch.npz
python demo_trajectory.py -t trajectory/wave.npz
```
