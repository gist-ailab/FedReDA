# print_param_stats.py
# DINOv2 + Reins 어댑터 구조에서
# - Backbone (non-adapter) params
# - Adapter1 (reins + linear_rein) params
# - Adapter2 (reins2 + linear_rein2) params
# 와 각 비율/통신량을 출력

import os
import sys
import argparse
import torch
import torch.nn as nn

# ----- repo 내부 경로 세팅 (FedDouble_2Adapter_ReinsOnly.py와 동일하게) -----
fednoro_path = './other_repos/FedNoRo'
if fednoro_path not in sys.path:
    sys.path.append(fednoro_path)

from dino_variant import _small_variant
from rein.models.backbones.reins_dinov2 import ReinsDinoVisionTransformer
from rein.models.backbones.reins import Reins


def build_model(num_classes: int, ckpt_path: str):
    """
    DINOv2 + Reins + head1/head2 구조를 그대로 만든 뒤
    state_dict를 로드한 모델을 반환 (CPU).
    """
    model = ReinsDinoVisionTransformer(**_small_variant)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)

    embed_dim = _small_variant['embed_dim']

    # 학생 어댑터 + head (Adapter1)
    model.reins = Reins(
        num_layers=_small_variant['depth'],
        embed_dims=embed_dim,
        patch_size=_small_variant['patch_size'],
    )
    model.linear_rein = nn.Linear(embed_dim, num_classes)

    # 교사 어댑터 + head (Adapter2)
    model.reins2 = Reins(
        num_layers=_small_variant['depth'],
        embed_dims=embed_dim,
        patch_size=_small_variant['patch_size'],
    )
    model.linear_rein2 = nn.Linear(embed_dim, num_classes)

    return model


def count_params(model: nn.Module):
    """
    파라미터를 세 가지로 나눠서 카운트:
      - backbone_params: 어댑터/헤드가 아닌 모든 파라미터
      - adapter1_params: 'reins.' / 'linear_rein' (학생)
      - adapter2_params: 'reins2.' / 'linear_rein2' (교사)
    """
    backbone_params = 0
    adapter1_params = 0
    adapter2_params = 0

    for name, p in model.named_parameters():
        n = p.numel()
        if name.startswith("reins2.") or name.startswith("linear_rein2"):
            adapter2_params += n
        elif name.startswith("reins.") or name.startswith("linear_rein"):
            adapter1_params += n
        else:
            backbone_params += n

    return backbone_params, adapter1_params, adapter2_params


def fmt(n):
    return f"{n:,d}"


def to_mb(n_params, bytes_per_param=4):
    return n_params * bytes_per_param / (1024 ** 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_classes",
        type=int,
        default=8,
        help="헤드 차원 (ISIC2019=8, ICH=5). 헤드 차이만 약간 생김.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./checkpoints/dinov2_vits14_pretrain.pth",
        help="DINOv2 사전학습 weight 경로",
    )
    parser.add_argument(
        "--bytes_per_param",
        type=int,
        default=4,
        help="통신량 계산 시 파라미터당 바이트 수 (FP32=4, FP16=2 등)",
    )
    args = parser.parse_args()

    model = build_model(args.num_classes, args.ckpt)
    backbone, adapter1, adapter2 = count_params(model)

    full_model_params = backbone + adapter1  # 'full model FL'에서 업데이트한다고 가정
    adapter_only_params = adapter1          # FedReDA에서 로컬에서 수정하는 파라미터

    print("========== Parameter statistics (DINOv2 + Reins) ==========")
    print(f"Num classes (head dim)         : {args.num_classes}")
    print()
    print(f"Backbone (non-adapter) params  : {fmt(backbone)} "
          f"({backbone/1e6:.2f}M, {to_mb(backbone, args.bytes_per_param):.2f} MB)")
    print(f"Adapter1 (reins + linear_rein) : {fmt(adapter1)} "
          f"({adapter1/1e6:.2f}M, {to_mb(adapter1, args.bytes_per_param):.2f} MB)")
    print(f"Adapter2 (reins2 + linear_rein2): {fmt(adapter2)} "
          f"({adapter2/1e6:.2f}M, {to_mb(adapter2, args.bytes_per_param):.2f} MB)")
    print()

    print("----- Ratios -----")
    print(f"Adapter1 / Backbone            : {adapter1 / backbone * 100:.2f}%")
    print(f"Adapter1 / (Backbone+Adapter1) : {adapter1 / full_model_params * 100:.2f}%")
    print()

    print("----- Communication per round per client (approx.) -----")
    up_params = adapter1            # FedReDA에서 업로드: 학생 어댑터 + head
    down_params = adapter2          # 다운로드: 교사 어댑터 + head (LOO or global)

    print(f"Upload (adapter1)   : {fmt(up_params)} params "
          f"(~{to_mb(up_params, args.bytes_per_param):.2f} MB)")
    print(f"Download (adapter2) : {fmt(down_params)} params "
          f"(~{to_mb(down_params, args.bytes_per_param):.2f} MB)")
    print()

    print("----- Hypothetical full-model FL -----")
    print(f"Trainable params if full model updated "
          f"(backbone + adapter1) : {fmt(full_model_params)} ({full_model_params/1e6:.2f}M)")
    print(f"Reduction (adapter-only vs full)      : "
          f"{(1 - adapter_only_params / full_model_params) * 100:.2f}% fewer trainable params")


if __name__ == "__main__":
    main()
