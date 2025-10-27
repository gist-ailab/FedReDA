import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from torch import Tensor

class Reins(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        query_dims: int = 256,
        token_length: int = 100,
        use_softmax: bool = True,
        link_token_to_query: bool = True,
        scale_init: float = 0.001,
        zero_mlp_delta_f: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.query_dims = query_dims
        self.token_length = token_length
        self.link_token_to_query = link_token_to_query
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.zero_mlp_delta_f = zero_mlp_delta_f
        self.create_model()

    def create_model(self):
        self.learnable_tokens = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims
            )
        )
        nn.init.uniform_(self.learnable_tokens.data, -val, val)
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))
        self.transform = nn.Linear(self.embed_dims, self.query_dims)
        self.merge = nn.Linear(self.query_dims * 3, self.query_dims)
        if self.zero_mlp_delta_f:
            del self.scale
            self.scale = 1.0
            nn.init.zeros_(self.mlp_delta_f.weight)
            nn.init.zeros_(self.mlp_delta_f.bias)

    def return_auto(self, feats):
        if self.link_token_to_query:
            tokens = self.transform(self.get_tokens(-1)).permute(1, 2, 0)
            tokens = torch.cat(
                [
                    F.max_pool1d(tokens, kernel_size=self.num_layers),
                    F.avg_pool1d(tokens, kernel_size=self.num_layers),
                    tokens[:, :, -1].unsqueeze(-1),
                ],
                dim=-1,
            )
            querys = self.merge(tokens.flatten(-2, -1))
            return feats, querys
        else:
            return feats

    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:
            # return all
            return self.learnable_tokens
        else:
            return self.learnable_tokens[layer]

    def forward(
        self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True
    ) -> Tensor:
        if batch_first:
            feats = feats.permute(1, 0, 2)
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)
        tokens = self.get_tokens(layer)
        delta_feat = self.forward_delta_feat(
            feats,
            tokens,
            layer,
        )
        delta_feat = delta_feat * self.scale
        feats = feats + delta_feat
        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)
        return feats

    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        attn = torch.einsum("nbc,mc->nbm", feats, tokens)
        if self.use_softmax:
            attn = attn * (self.embed_dims**-0.5)
            attn = F.softmax(attn, dim=-1)
        delta_f = torch.einsum(
            "nbm,mc->nbc",
            attn[:, :, 1:],
            self.mlp_token2feat(tokens[1:, :]),
        )
        delta_f = self.mlp_delta_f(delta_f + feats)
        return delta_f
    
    # def forward_delta_only(self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True) -> Tensor:
    #     if batch_first:
    #         feats = feats.permute(1, 0, 2)
    #     if has_cls_token:
    #         _, feats = torch.tensor_split(feats, [1], dim=0)
    #     tokens = self.get_tokens(layer)
    #     delta_feat = self.forward_delta_feat(feats, tokens, layer)
    #     delta_feat = delta_feat * self.scale
    #     if batch_first:
    #         delta_feat = delta_feat.permute(1, 0, 2)
    #     return delta_feat
    
    def forward_delta_only(self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True) -> Tensor:
        if batch_first:
            feats = feats.permute(1, 0, 2)
        cls_token = None
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)
        tokens = self.get_tokens(layer)
        delta_feat = self.forward_delta_feat(feats, tokens, layer)
        delta_feat = delta_feat * self.scale
        if has_cls_token:
            delta_feat = torch.cat([torch.zeros_like(cls_token), delta_feat], dim=0)
        if batch_first:
            delta_feat = delta_feat.permute(1, 0, 2)
        return delta_feat
    
class DynamicReins(nn.Module):
    """
    기존 Rein의 down->act->up 잔차 경로는 유지.
    차이: r_active만큼만 사용하도록 mask 적용. 파라미터 shape는 그대로라서 기존 체크포인트와 호환.
    """
    def __init__(self, dim, r_max: int, alpha: float = 1.0, dropout: float = 0.0, act=nn.GELU, pre_norm=True):
        super().__init__()
        self.dim = dim
        self.r_max = int(r_max)
        self.r_active = int(r_max)   # 동적 변경 대상
        self.alpha = alpha

        self.pre_norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()
        self.down = nn.Linear(dim, r_max, bias=False)
        self.up   = nn.Linear(r_max, dim, bias=False)
        self.act = act()
        self.drop = nn.Dropout(dropout)

        # 학습 가능한 residual gate(안정화)
        self.gate = nn.Parameter(torch.tensor(1.0))

        # 앞 r_active 개만 활성화하는 마스크 버퍼
        self.register_buffer("mask", torch.ones(r_max))  # [r_max]

        # 수치 안정화용 작은 스케일(필요 시 사용)
        self.register_buffer("eps", torch.tensor(1e-6))
        
        self._attach_grad_hooks()

    @torch.no_grad()
    def freeze_inactive(self):
        r = self.r_active
        self.down.weight[r:, :].zero_()
        self.up.weight[:, r:].zero_()
    
    @torch.no_grad()
    def set_rank(self, r: int):
        r = int(max(1, min(r, self.r_max)))
        if r > self.r_active:
            import math, torch.nn as nn
            nn.init.kaiming_uniform_(self.down.weight[self.r_active:r, :], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.up.weight[:, self.r_active:r], a=math.sqrt(5))
        self.r_active = r

    def get_rank(self) -> int:
        return int(self.r_active)

    def _masked_linear_down(self, x):
        # Wd': [r_max, dim] where tail rows are zeroed
        Wd = self.down.weight * self.mask.view(-1, 1)
        return F.linear(x, Wd)  # [*, r_max]지만 tail이 0

    def _masked_linear_up(self, z):
        # Wu': [dim, r_max] where tail cols are zeroed
        Wu = self.up.weight * self.mask.view(1, -1)
        return F.linear(z, Wu)  # [*, dim]

    def forward(self, x, layer=None, batch_first=True, has_cls_token=True):
        # 입력이 (B, T, C)라는 가정. batch_first=False로 오면 (T, B, C) → (B, T, C)로 변환
        transposed = False
        if not batch_first:
            x = x.permute(1, 0, 2)  # (T,B,C) -> (B,T,C)
            transposed = True

        h = self.pre_norm(x)
        r = self.r_active
        z = F.linear(h, self.down.weight[:r, :])   # (B,T,C) x (r, C) -> (B,T,r)
        z = self.act(z); z = self.drop(z)
        z = F.linear(z, self.up.weight[:, :r])     # (B,T,r) x (C, r) -> (B,T,C)
        out = x + self.gate * self.alpha * z

        if transposed:
            out = out.permute(1, 0, 2)  # (B,T,C) -> (T,B,C)
        return out
    
    # 2) grad 훅(초기화 시 1회 등록)
    def _attach_grad_hooks(self):
        def hook_down(g):
            r = self.r_active
            mask = g.new_zeros(g.shape); mask[:r, :] = 1
            return g * mask
        def hook_up(g):
            r = self.r_active
            mask = g.new_zeros(g.shape); mask[:, :r] = 1
            return g * mask
        self.down.weight.register_hook(hook_down)
        self.up.weight.register_hook(hook_up)