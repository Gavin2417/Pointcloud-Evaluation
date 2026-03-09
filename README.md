# HyPER

A robust, hybrid autonomous navigation framework that integrates geometric heuristics (STEP) with deep-learning semantic segmentation (RandLA-Net) for safety-critical risk assessment in unstructured, 3D subterranean environments.

---

## Video Demonstrations

### HyPER — Full System
[![HyPER Demo](https://img.youtube.com/vi/zdk7TQ-G3aY/maxresdefault.jpg)](https://youtu.be/zdk7TQ-G3aY)

### STEP — Geometric Heuristics
[![STEP Demo](https://img.youtube.com/vi/mU_K56Iut88/maxresdefault.jpg)](https://youtu.be/mU_K56Iut88)

### RandLA-Net — Semantic Segmentation
[![RandLA-Net Demo](https://img.youtube.com/vi/VYXeMFNJ6K8/maxresdefault.jpg)](https://youtu.be/VYXeMFNJ6K8)

### Simple Fusion — Baseline Comparison
[![Simple Fusion Demo](https://img.youtube.com/vi/1gkk3FRGyj8/maxresdefault.jpg)](https://youtu.be/1gkk3FRGyj8)

---

## Overview

HyPER addresses the challenge of safe autonomous navigation in GPS-denied, unstructured subterranean environments by fusing two complementary risk estimation strategies:

| Component | Type | Role |
|-----------|------|------|
| STEP | Geometric | Interpretable traversability heuristics |
| RandLA-Net | Deep Learning | Fast semantic segmentation of 3D point clouds |
| HyPER Fusion | Hybrid | Confidence-weighted risk map integration |

---

## Repository Structure

```
hyper/
├── step/               # Geometric heuristics module
├── randlanet/          # Semantic segmentation module
├── fusion/             # HyPER hybrid fusion layer
├── evaluation/         # Benchmarking and comparison tools
└── configs/            # Experiment configurations
```

---

## License

This project is released under the [MIT License](LICENSE).
