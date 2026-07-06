# mujoco-sim

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  [![Release](https://img.shields.io/github/v/release/wuji-technology/mujoco-sim)](https://github.com/wuji-technology/mujoco-sim/releases)

Simulation demo for MuJoCo. This repository provides a minimal example for loading and controlling the Wuji Hand in MuJoCo simulator. Loads MJCF models from the description submodule and plays pre-recorded trajectories in a loop.

**Get started with [Quick Start](#quick-start). For detailed documentation, please refer to [MuJoCo Simulation Example](https://docs.wuji.tech/docs/en/wuji-description/latest/related-repos/#41-mujoco-simulation-example) on Wuji Docs Center.**

https://github.com/user-attachments/assets/4b3d6d5c-420e-4e15-bbe7-68bcad9729f0

<video src="./assets/video.mp4" controls=""></video>

## Repository Structure

```text
├── assets/                    // Demo video files
│   └── video.mp4
├── data/                      // Pre-recorded trajectory data
│   └── wave.npy
├── wuji_hand_description/     // Submodule: hand model (MJCF, meshes)
├── run_sim.py                 // Main simulation script
├── requirements.txt           // Python dependencies
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.8+

### Installation

```bash
git clone --recursive https://github.com/wuji-technology/mujoco-sim.git
cd mujoco-sim
pip install -r requirements.txt
```

### Running

```bash
python run_sim.py
```

The script loads the default right-hand model and plays the trajectory from `data/wave.npy` in a loop. To use the left hand, edit `side = "left"` in `run_sim.py`.

### Update Models

To update hand models to the latest version:

```bash
git submodule update --remote
```

## Contact

For any questions, please contact [support@wuji.tech](mailto:support@wuji.tech).
