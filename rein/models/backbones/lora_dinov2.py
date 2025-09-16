from .lora import _LoRA_qkv
from .dino_v2 import DinoVisionTransformer
from .utils import set_requires_grad, set_train
import torch
import torch.nn as nn

class LoRADinoVisionTransformer(DinoVisionTransformer):
    def __init__(self, r=4, alpha=16, lora_layers=None, **kwargs):
        super().__init__(**kwargs)
        self.lora_layers = lora_layers if lora_layers else list(range(len(self.blocks)))
        self.lora_modules = {}

        for idx, blk in enumerate(self.blocks):
            if idx in self.lora_layers:
                orig_qkv = blk.attn.qkv
                dim = orig_qkv.in_features
                w_a_q = nn.Linear(dim, r, bias=False)
                w_b_q = nn.Linear(r, dim, bias=False)
                w_a_k = nn.Linear(dim, r, bias=False)
                w_b_k = nn.Linear(r, dim, bias=False)
                w_a_v = nn.Linear(dim, r, bias=False)
                w_b_v = nn.Linear(r, dim, bias=False)
                blk.attn.qkv = _LoRA_qkv(
                    orig_qkv, w_a_q, w_b_q,
                    w_a_k, w_b_k,
                    w_a_v, w_b_v,
                    r, alpha
                )
                self.lora_modules[idx] = blk.attn.qkv

    def forward_features(self, x, masks=None, use_lora=True, return_both=False):
        """
        use_lora=False → LoRA 비적용 출력
        return_both=True → (base, lora_delta, combined) 출력 리스트 반환
        """
        outputs = [] if return_both else None
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            if idx in self.lora_modules:
                self.lora_modules[idx].use_lora = use_lora
                if return_both:
                    base, delta, combined = self.lora_modules[idx](x, return_both=True)
                    outputs.append((base, delta, combined))
                    x = combined
                else:
                    x = blk(x)
            else:
                x = blk(x)
        return (x, outputs) if return_both else x

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)

        for p in self.parameters():
            p.requires_grad = False

        for idx, blk in enumerate(self.blocks):
            if hasattr(blk.attn, "qkv") and isinstance(blk.attn.qkv, _LoRA_qkv):
                for p in blk.attn.qkv.parameters():
                    p.requires_grad = True

        if hasattr(self, "head"):
            for p in self.head.parameters():
                p.requires_grad = True
        if hasattr(self, "linear"):
            for p in self.linear.parameters():
                p.requires_grad = True

        super().train(mode)
        
        for m in self.modules():
            if any(p.requires_grad for p in m.parameters(recurse=False)):
                m.train(mode)
        return self