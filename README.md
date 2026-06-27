# 🚀 Project Mars

A simulation environment for autonomous Mars rover navigation and dataset generation using NVIDIA Isaac Sim.

This project focuses on collecting high-quality visual datasets for training Vision-Language-Action (VLA) models that enable autonomous rover navigation on Mars-like terrain.

---

## Overview

Project Mars simulates a planetary exploration rover inside NVIDIA Isaac Sim. The rover autonomously explores a procedurally generated Mars environment while recording observations that can later be used for machine learning.

The long-term objective is to train a VLA model capable of understanding natural language commands such as:

- "Go to the door"
- "Drive towards the large rock"
- "Move around the crater"
- "Navigate to the charging station"

using only RGB camera observations.

---

## Features

- Autonomous rover exploration
- Randomized navigation policy
- Mars terrain simulation
- RGB image collection
- Episode recording
- Domain randomization
- Automatic dataset generation
- Isaac Sim integration
- Ready for VLA training pipelines

---

## Repository Structure

```
Project_Mars/
│
├── assets/               # Rover assets and environments
├── configs/              # Simulation configuration
├── scripts/              # Simulation scripts
├── policies/             # Expert policy
├── datasets/             # Generated datasets
├── models/               # Trained models
├── logs/                 # Simulation logs
└── README.md
```

*(Folder names can be updated to match the repository.)*

---

## Requirements

- NVIDIA Isaac Sim
- Python 3.11+
- CUDA-enabled NVIDIA GPU
- RTX Rendering Enabled
- Ubuntu 22.04 or Windows 11
- NVIDIA Omniverse

---

## Installation

Clone the repository

```bash
git clone https://github.com/priyan212/Project_Mars.git
```

Navigate into the project

```bash
cd Project_Mars
```

Launch Isaac Sim and open the project.

---

## Dataset Generation Pipeline

```
Mars Environment
        │
        ▼
 Autonomous Rover
        │
        ▼
 Expert Navigation Policy
        │
        ▼
RGB Camera Capture
        │
        ▼
Episode Recording
        │
        ▼
Dataset Storage
        │
        ▼
VLA Model Training
```

---

## Current Goal

The current objective is to build an expert policy capable of:

- Exploring the environment autonomously
- Avoiding obstacles
- Visiting diverse locations
- Maximizing scene coverage
- Recording high-quality RGB observations

The collected data will later be used for imitation learning and Vision-Language-Action model training.

---

## Planned Features

- [ ] Autonomous expert policy
- [ ] Waypoint planner
- [ ] Coverage-aware exploration
- [ ] Terrain randomization
- [ ] Lighting randomization
- [ ] Weather randomization
- [ ] Automatic episode recording
- [ ] COCO dataset export
- [ ] Segmentation dataset generation
- [ ] VLA dataset export
- [ ] ROS2 integration
- [ ] Sim-to-Real pipeline

---

## Technologies

- NVIDIA Isaac Sim
- Omniverse
- Python
- PyTorch
- OpenCV
- NumPy
- Isaac Lab
- ROS2 (planned)

---

## Future Work

The final vision of Project Mars is an end-to-end autonomous Mars rover capable of understanding natural language instructions and safely navigating unknown environments using only onboard vision.

---

## Contributing

Contributions are welcome.

Feel free to open an issue or submit a pull request for improvements, bug fixes, or new features.

---

## License

This project is licensed under the MIT License.

---

## Author

**Priyan Malakar**

GitHub: https://github.com/priyan212
