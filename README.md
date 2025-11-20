# FedReDA: Federated Reliability-aware Dual Adapters for Noisy-Label Learning on Vision Foundation Models

This repository contains the reference implementation of **FedReDA**,  
a federated noisy-label learning method built on a frozen DINOv2 backbone with  
dual Reins adapters (student/teacher) and noise-aware distillation.

## Features

- Frozen **DINOv2 ViT-S/14** backbone (`_small_variant`)
- Two Reins adapters:
  - `reins`  : per-client student adapter (Adapter1)
  - `reins2` : global / LOO teacher adapter (Adapter2)
- **LOO teacher** or shared FedAvg teacher
- **GMM + agreement mask** 기반 clean/noisy 샘플 구분
- **Noisy KD** + clean CE + FedProx-style regularization
- **ComputeTracker** 로 GPU+CPU 시간 및 샘플 수 자동 로깅

---

## Repository Structure (예시)

```text
.
├── FedReDA.py
├── dino_variant.py
├── other_repos/
│   └── FedNoRo/
├── rein/
│   └── models/backbones/
│       ├── reins_dinov2.py
│       └── reins.py
├── dataset/
│   └── dataset.py
├── utils/
│   └── utils.py
└── checkpoints/
    └── dinov2_vits14_pretrain.pth


