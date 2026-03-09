# HyPER

A robust, hybrid autonomous navigation framework that integrates geometric heuristics (STEP) with deep-learning semantic segmentation (RandLA-Net) for safety-critical risk assessment in unstructured, 3D subterranean environments.

---

This research makes use of:

- [RandLA-Net PyTorch](https://github.com/tsunghan-wu/RandLA-Net-pytorch)
- [Cosys-AirSim](https://github.com/Cosys-Lab/Cosys-AirSim) with UE block environment

## Overview

HyPER addresses the challenge of safe autonomous navigation in GPS-denied, unstructured subterranean environments by fusing two complementary risk estimation strategies:

| Component     | Type          | Role                                          |
|---------------|---------------|-----------------------------------------------|
| STEP          | Geometric     | Interpretable traversability heuristics       |
| RandLA-Net    | Deep Learning | Fast semantic segmentation of 3D point clouds |
| Simple Fusion | Baseline      | Naive sensor fusion for benchmarking          |
| HyPER Fusion  | Hybrid        | Confidence-weighted risk map integration      |

## Repository Structure

```text
HyPER/
├── Frameworks/           # Geometric heuristics (STEP) and related modules
├── Frameworks/rand/      # RandLA-Net semantic segmentation implementation
├── Frameworks/record/    # Experiment records, statistics, and logs
├── configs/              # Experiment configurations
└── environment.yml       # Conda environment for RandLA-Net and Cosys-AirSim
```

## Video Demonstrations

### STEP — Geometric Heuristics
[![STEP Demo](https://img.youtube.com/vi/mU_K56Iut88/maxresdefault.jpg)](https://youtu.be/mU_K56Iut88)

### RandLA-Net — Semantic Segmentation
[![RandLA-Net Demo](https://img.youtube.com/vi/VYXeMFNJ6K8/maxresdefault.jpg)](https://youtu.be/VYXeMFNJ6K8)

### Simple Fusion — Baseline Comparison
[![Simple Fusion Demo](https://img.youtube.com/vi/1gkk3FRGyj8/maxresdefault.jpg)](https://youtu.be/1gkk3FRGyj8)

### HyPER — Full System
[![HyPER Demo](https://img.youtube.com/vi/zdk7TQ-G3aY/maxresdefault.jpg)](https://youtu.be/zdk7TQ-G3aY)

---

## License

This project is released under the [MIT License](LICENSE).
