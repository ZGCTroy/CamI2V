import pdb

import numpy as np
import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from lvdm.common import checkpoint, default


def normalize(pts, H, W):
    """
    Args:
        pts: *N x 2 ([0~W-1,0~H-1] normalized to [-1,1])
    """
    pts[..., 0] = -1.0 + 2.0 * pts[..., 0] / (W - 1)
    pts[..., 1] = -1.0 + 2.0 * pts[..., 1] / (H - 1)

    return pts


def de_normalize(pts, H, W):
    """
    Args:
        pts: *N x 2 ([-1,1] denormalized to [0~W-1,0~H-1] )
    """
    WH = torch.tensor([[W, H]], dtype=pts.dtype, device=pts.device)
    return (pts + 1) * (WH - 1) / 2.0


def pix2coord(x, downsample):
    """convert pixels indices to real coordinates for 3D 2D projection"""
    return x * downsample + downsample / 2.0 - 0.5


def coord2pix(y, downsample):
    """convert real coordinates to pixels indices for 3D 2D projection"""
    # x * downsample + downsample / 2.0 - 0.5 = y
    return (y + 0.5 - downsample / 2.0) / downsample


class EpipolarCrossAttention(nn.Module):

    def __init__(self, query_dim, context_dim=None, out_dim=None, heads=8, dim_head=64,
                 dropout=0.0, num_register_tokens=0):
        super().__init__()
        inner_dim = dim_head * heads
        self.context_dim = context_dim
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.dropout = dropout
        self.out_dim = out_dim
        if out_dim is not None:
            self.to_out = nn.Sequential(nn.Linear(inner_dim, out_dim), nn.Dropout(dropout))
        else:
            self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

        self.num_register_tokens = num_register_tokens

        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.randn((1, num_register_tokens, context_dim)), requires_grad=True)

    def forward(self, x: Tensor, context: Tensor, attn_mask: Tensor = None):
        '''
        :param x:       B,L1,C
        :param context:       B,L2,C
        :param attn_mask: B,L1,L2
        :return:
        '''
        # pdb.set_trace()
        q = self.to_q(x)
        B = q.shape[0]

        if self.num_register_tokens > 0:
            context = torch.concat([self.register_tokens.repeat(B, 1, 1), context], dim=1)  # B, L2, D --> B, num_registers+L2, D

        k = self.to_k(context)
        v = self.to_v(context)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask = torch.nn.functional.pad(attn_mask, (self.num_register_tokens,0), mode='constant', value=True) # B,L1,L2 --> B,L1, num_registers+L2
            else:
                attn_mask = torch.nn.functional.pad(attn_mask, (self.num_register_tokens,0), mode='constant', value=-0.0) # B,L1,L2 --> B,L1, num_registers+L2

        q, k, v = map(lambda t: rearrange(t, "B L (H D) -> B H L D", H=self.heads), (q, k, v))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask.unsqueeze(1) if attn_mask is not None else None)
        out = rearrange(out, "B H L D -> B L (H D)")

        return self.to_out(out)


class Epipolar(nn.Module):
    def __init__(self, query_dim, context_dim, heads, origin_h=256, origin_w=256,
                 is_3d_full_attn=False, num_register_tokens=0, compression_factor=1, attention_resolution=[8, 4, 2, 1],
                 only_on_cond_frame=False, **kwargs):
        super(Epipolar, self).__init__()
        self.attention_resolution = attention_resolution
        self.origin_h = origin_h
        self.origin_w = origin_w
        self.num_heads = heads
        self.is_3d_full_attn = is_3d_full_attn
        self.only_on_cond_frame = only_on_cond_frame
        self.compression_factor = compression_factor
        self.epipolar_attn = EpipolarCrossAttention(
            query_dim=query_dim,
            context_dim=context_dim,
            out_dim=None,
            heads=heads,
            dim_head=int(query_dim // heads // self.compression_factor),
            num_register_tokens=num_register_tokens,
        )
        nn.init.zeros_(list(self.epipolar_attn.to_out[0].parameters())[0])
        nn.init.zeros_(list(self.epipolar_attn.to_out[0].parameters())[1])

    def forward(self, features: Tensor, sample_locs_dict: dict[int, Tensor] = None, cond_frame_index=None, **kwargs):
        """
        Args:
            features: B x T x C x H x W
            sample_locs_dict: {8, 16, 32, 64} -> B x L1=THW x L2=THW
        """
        B, T1, C, H, W = features.shape

        if sample_locs_dict is not None and not self.is_3d_full_attn:
            with torch.no_grad():
                attn_mask = sample_locs_dict.get(self.origin_h // H, None)
        else:
            attn_mask = None

        x = rearrange(features, "B T1 C H W -> B (T1 H W) C")
        if not self.only_on_cond_frame:
            context = x
        else:
            assert cond_frame_index is not None
            context = rearrange(
                features[torch.arange(B, device=x.device), cond_frame_index, ...].unsqueeze(1),
                "B T1 C H W -> B (T1 H W) C"
            )
            if attn_mask is not None:
                attn_mask = rearrange(attn_mask, "B L1 (T2 H W) -> B L1 T2 (H W)", H=H, W=W)
                attn_mask = attn_mask[torch.arange(B, device=x.device), :, cond_frame_index, :] # B L1 T2 (H W) -> B L1 L2=(H W)

        out = self.epipolar_attn(x, context=context, attn_mask=attn_mask)

        return rearrange(out, "B (T1 H W) C -> (B H W) T1 C", B=B, T1=T1, C=C, H=H, W=W)



