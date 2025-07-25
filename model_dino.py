import torch
import torch.nn as nn
import torch.nn.functional as F
import dino_variant
from rein import LoRAFusedDualReinsDinoVisionTransformer, ReinsDinoVisionTransformer

class LoRAFusedReinDinov2():
    def __init__(self, variant, dino_state_dict, num_classes):
        super().__init__()
        self.backbone = LoRAFusedDualReinsDinoVisionTransformer(**variant)
        self.backbone.load_state_dict(dino_state_dict, strict=False)
        self.backbone.linear_rein = nn.Linear(variant['embed_dim'], num_classes)
        self.fusion_alpha = self.backbone.fusion_alpha  # Shortcut

    def forward(self, x):
        feats = self.backbone.forward_fused_features(x)
        logits = self.backbone.linear_rein(feats[:, 0, :])
        return logits

    def train1(self):
        self.backbone.train1()

    def train2(self):
        self.backbone.train2()

    def freeze_shared_adapter(self, global_model):
        self.backbone.reins2.load_state_dict(global_model.reins.state_dict())
        for p in self.backbone.reins2.parameters():
            p.requires_grad = False
        self.backbone.reins2.eval()

    def get_backbone(self):
        return self.backbone
    
class ReinDinov2(nn.Module):
    def __init__(self, variant, dino_state_dict, num_classes):
        super().__init__()
        self.backbone = ReinsDinoVisionTransformer(**variant)
        self.backbone.load_state_dict(dino_state_dict, strict=False)
        
        self.backbone.linear_rein = nn.Linear(variant['embed_dim'], num_classes)

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        logits = self.backbone.linear_rein(feats[:, 0, :])
        return logits
    
    def forward_wfeat(self, x):
        feats = self.backbone.forward_features(x)
        logits = self.backbone.linear_rein(feats[:, 0, :])
        return logits, feats[:, 0, :]
    
    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.train(mode)
        return self

    def get_backbone(self):
        return self.backbone
    
class ReinDinov2_trans(nn.Module):
    def __init__(self, variant, dino_state_dict, num_classes):
        super().__init__()
        self.backbone = ReinsDinoVisionTransformer(**variant)
        self.backbone.load_state_dict(dino_state_dict, strict=False)
        self.backbone.bayes_linear = nn.Linear(variant['embed_dim'], num_classes)

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        logits = self.backbone.bayes_linear(feats[:, 0, :])
        logits = logits.reshape(logits.size(0),7,7)
        logits = F.softmax(logits, dim=2)
        return logits
    
    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.train(mode)
        return self

    def get_backbone(self):
        return self.backbone