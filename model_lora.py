import torch
import torch.nn as nn
import torch.nn.functional as F
import dino_variant
from rein import LoRADinoVisionTransformer

class LoRADinov2(nn.Module):
    def __init__(self, variant, dino_state_dict, num_classes):
        super().__init__()
        self.backbone = LoRADinoVisionTransformer(**variant)
        self.backbone.load_state_dict(dino_state_dict, strict=False)
        
        self.backbone.linear_rein = nn.Linear(variant['embed_dim'], num_classes)

    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[NaN detected] Input contains NaN/Inf. shape={x.shape}, "
                  f"min={x.min().item()}, max={x.max().item()}")
        feats = self.backbone.forward_features(x)
        if torch.isnan(feats).any() or torch.isinf(feats).any():
            print(f"[NaN detected] feats contain NaN/Inf. "
                  f"min={feats.min().item()}, max={feats.max().item()}")
        logits = self.backbone.linear_rein(feats[:, 0, :])
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"[NaN detected] logits contain NaN/Inf. "
                  f"min={logits.min().item()}, max={logits.max().item()}")
        return logits
    
    def forward_wfeat(self, x):
        feats = self.backbone.forward_features(x)
        logits = self.backbone.linear_rein(feats[:, 0, :])
        return logits, feats[:, 0, :]
    
    def forward_nolora(self, x):
        feats = self.backbone.forward_features(x)
    
    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.train(mode)
        return self

    def get_backbone(self):
        return self.backbone