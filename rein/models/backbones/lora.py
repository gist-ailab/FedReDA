import os
import json
import torch
import numpy as np
import math

import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from torch.nn.parameter import Parameter

class _LoRA_qkv(nn.Module):
    """ https://github.com/meiluzhu/DEeR/blob/main/models_dict/CLIP_DyLoRA.py
    In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """
    def __init__(self, qkv, 
                 linear_a_q, linear_b_q,
                 linear_a_k, linear_b_k,
                 linear_a_v, linear_b_v,
                 r, alpha):
        super().__init__()
        self.qkv = qkv              # 원본 qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_k = linear_a_k
        self.linear_b_k = linear_b_k
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.scaling = alpha / r
        self.use_lora = True        # 기본값: LoRA 사용

    def forward(self, x, return_both=False):
        """return_both=True면 (base, lora, combined) 반환"""
        base_qkv = self.qkv(x)  # 원본 qkv 출력

        if not self.use_lora and not return_both:
            return base_qkv

        # LoRA 델타 계산
        def delta(a, b):  # a,b: Linear
            return (x @ a.weight.T @ b.weight.T) * self.scaling

        dq = delta(self.linear_a_q, self.linear_b_q)
        dk = delta(self.linear_a_k, self.linear_b_k)
        dv = delta(self.linear_a_v, self.linear_b_v)

        lora_qkv = torch.zeros_like(base_qkv)
        lora_qkv[:, :, :self.dim] = dq
        lora_qkv[:, :, self.dim:-self.dim] = dk
        lora_qkv[:, :, -self.dim:] = dv

        combined = base_qkv + lora_qkv

        if return_both:
            return base_qkv, lora_qkv, combined
        else:
            return combined

def set_trainable_params(model):
    for n, p in model.parameters():
        if 'lora_' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False