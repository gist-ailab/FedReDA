from .reins import Reins, DynamicReins
from .dino_v2 import DinoVisionTransformer
from .utils import set_requires_grad, set_train
import torch.nn as nn
import torch


class ReinsDinoVisionTransformer(DinoVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        adapter_kind = 'rein'
        self.adapter_kind = adapter_kind
        
        if adapter_kind == "rein":
            # 원본 Reins(토큰 기반)
            self.reins = Reins(
                num_layers=kwargs['depth'],
                embed_dims=kwargs['embed_dim'],
                patch_size=kwargs['patch_size'],
            )
            self._has_dynamic = False
        else:
            # DynamicReins(저랭크 bottleneck)
            rein_r_max=64
            rein_alpha=1.0
            rein_pre_norm=True
            self.reins = DynamicReins(
                dim=kwargs['embed_dim'],
                r_max=rein_r_max,
                alpha=rein_alpha,
                pre_norm=rein_pre_norm,
            )
            self._has_dynamic = True

        # 분류기(리니어 헤드)
        # self.linear_rein = nn.Linear(kwargs['embed_dim'], kwargs.get('num_classes', 5))

    # 동적 rank 제어(있으면 호출, 없으면 무시)
    def set_rank_all(self, r: int):
        if self._has_dynamic and hasattr(self.reins, "set_rank"):
            self.reins.set_rank(int(r))

    # 고정 rank로 “잠그기”: 동적 모듈이어도 r을 고정하고 동결
    def lock_rank(self, r: int):
        if self._has_dynamic and hasattr(self.reins, "set_rank"):
            self.reins.set_rank(int(r))
            # 비활성 영역 0 고정
            if hasattr(self.reins, "freeze_inactive"):
                self.reins.freeze_inactive()

    def forward_features(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins.forward(x, idx, batch_first=True, has_cls_token=True)
        return x

    def forward_features_widx(self, x, idxs, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx in idxs:
                x = self.reins.forward(x, idx, batch_first=True, has_cls_token=True)
        return x

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins", "linear"])
        set_train(self, ["reins", "linear"])
        
class SelectiveReinsDinoVisionTransformer(DinoVisionTransformer):
    def __init__(self, rein_layers=None, **kwargs):
        """
        Args:
            rein_layers (list): Rein 어댑터를 적용할 layer 인덱스 리스트
        """
        super().__init__(**kwargs)
        self.rein_layers = rein_layers if rein_layers is not None else []
        self.reins = Reins(
            num_layers=kwargs['depth'],
            embed_dims=kwargs['embed_dim'],
            patch_size=kwargs['patch_size'],
        )

    def forward_features(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx in self.rein_layers:
                x = self.reins.forward(x, idx, batch_first=True, has_cls_token=True)
        return x
    
    def forward_features2(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx in self.rein_layers:
                x = self.reins2.forward(x, idx, batch_first=True, has_cls_token=True)
        return x

    def forward_features_no_rein(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for blk in self.blocks:
            x = blk(x)
        return x

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins", "linear"])
        set_train(self, ["reins", "linear"])

class DualReinsDinoVisionTransformer(DinoVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reins1 = Reins(
            num_layers=kwargs['depth'],
            embed_dims=kwargs['embed_dim'],
            patch_size=kwargs['patch_size'],
        )
        self.reins2 = Reins(
            num_layers=kwargs['depth'],
            embed_dims=kwargs['embed_dim'],
            patch_size=kwargs['patch_size'],
        )

    def forward_features1(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins1.forward(x, idx, batch_first=True, has_cls_token=True)
        return x

    def forward_features2(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins2.forward(x, idx, batch_first=True, has_cls_token=True)
        return x

    def forward_features_no_rein(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for blk in self.blocks:
            x = blk(x)
        return x

    def train1(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins1", "linear"])
        set_train(self, ["reins1", "linear"])

    def train2(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins2", "linear"])
        set_train(self, ["reins2", "linear"])

    def train_all(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins1", "reins2", "linear"])
        set_train(self, ["reins1", "reins2", "linear"])

class LoRAFusedDualReinsDinoVisionTransformer(DinoVisionTransformer):
    def __init__(self, fusion_alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.reins1 = Reins(
            num_layers=kwargs['depth'],
            embed_dims=kwargs['embed_dim'],
            patch_size=kwargs['patch_size'],
        )
        self.reins2 = Reins(
            num_layers=kwargs['depth'],
            embed_dims=kwargs['embed_dim'],
            patch_size=kwargs['patch_size'],
        )
        self.fusion_alpha = nn.Parameter(torch.tensor(fusion_alpha), requires_grad=True)

    def forward_features(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins1.forward(x, idx, batch_first=True, has_cls_token=True)
        return x

    def forward_features1(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins1.forward(x, idx, batch_first=True, has_cls_token=True)
        return x

    def forward_features2(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins2.forward(x, idx, batch_first=True, has_cls_token=True)
        return x

    def forward_fused_features(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            delta1 = self.reins1.forward_delta_only(x, idx, batch_first=True, has_cls_token=True)
            delta2 = self.reins2.forward_delta_only(x, idx, batch_first=True, has_cls_token=True)
            x = x + self.fusion_alpha * delta1 + (1 - self.fusion_alpha) * delta2
            # x = x + delta1 + delta2
        return x
    
    def forward_features1_wfeats(self, x, masks=None):
        features = []
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins1.forward(x, idx, batch_first=True, has_cls_token=True)
            features.append(x)
        return features

    def forward_features2_wfeats(self, x, masks=None):
        features = []
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins2.forward(x, idx, batch_first=True, has_cls_token=True)
            features.append(x)
        return features
    
    def forward_fused_features_wfeats(self, x, masks=None):
        features = []
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            delta1 = self.reins1.forward_delta_only(x, idx, batch_first=True, has_cls_token=True)
            delta2 = self.reins2.forward_delta_only(x, idx, batch_first=True, has_cls_token=True)
            # x = x + self.fusion_alpha * delta1 + (1 - self.fusion_alpha) * delta2
            x = x + delta1 + delta2
            features.append(x)
        return features

    def forward_features_no_rein(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for blk in self.blocks:
            x = blk(x)
        return x

    def train1(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins1", "linear"])
        set_train(self, ["reins1", "linear"])

    def train2(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins2", "linear"])
        set_train(self, ["reins2", "linear"])

    def train_all(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins1", "reins2", "linear"])
        set_train(self, ["reins1", "reins2", "linear"])

