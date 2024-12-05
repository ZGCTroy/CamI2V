import logging
import pdb

import torch
from einops import rearrange, repeat

from lvdm.models.utils_diffusion import timestep_embedding
from lvdm.modules.attention import SpatialTransformer, TemporalTransformer
from lvdm.modules.networks.openaimodel3d import TimestepBlock
from functools import partial
from lvdm.common import (
    checkpoint,
    exists,
    default,
)
import math
try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

mainlogger = logging.getLogger('mainlogger')


# add camera_condition input to forward of unet
def new_forward_for_unet(self, x, timesteps, context=None, features_adapter=None, fs=None, camera_condition=None, **kwargs):
    b, _, t, _, _ = x.shape
    t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).type(x.dtype)
    emb = self.time_embed(t_emb)

    ## repeat t times for context [(b t) 77 768] & time embedding
    ## check if we use per-frame image conditioning
    _, l_context, _ = context.shape
    if l_context == 77 + t * 16:  ## !!! HARD CODE here                     # interp_mode
        context_text, context_img = context[:, :77, :], context[:, 77:, :]
        context_text = context_text.repeat_interleave(repeats=t, dim=0)
        context_img = rearrange(context_img, 'b (t l) c -> (b t) l c', t=t)
        context = torch.cat([context_text, context_img], dim=1)
    else:
        context = context.repeat_interleave(repeats=t, dim=0)           # single cond frame
    emb = emb.repeat_interleave(repeats=t, dim=0)

    ## always in shape (b t) c h w, except for temporal layer
    x = rearrange(x, 'b c t h w -> (b t) c h w')

    ## combine emb
    if self.fs_condition:
        if fs is None:
            fs = torch.tensor(
                [self.default_fs] * b, dtype=torch.long, device=x.device)
        fs_emb = timestep_embedding(fs, self.model_channels, repeat_only=False).type(x.dtype)

        fs_embed = self.fps_embedding(fs_emb)
        fs_embed = fs_embed.repeat_interleave(repeats=t, dim=0)
        emb = emb + fs_embed

    h = x.type(self.dtype)
    adapter_idx = 0
    hs = []
    for id, module in enumerate(self.input_blocks):
        ########################################### only change here, add camera_condition input ###########################################
        h = module(h, emb, context=context, batch_size=b, camera_condition=camera_condition)
        ########################################### only change here, add camera_condition input ###########################################
        if id == 0 and self.addition_attention:
            ########################################### only change here, add camera_condition input ###########################################
            h = self.init_attn(h, emb, context=context, batch_size=b, camera_condition=camera_condition)
            ########################################### only change here, add camera_condition input ###########################################
        ## plug-in adapter features
        if ((id + 1) % 3 == 0) and features_adapter is not None:
            h = h + features_adapter[adapter_idx]
            adapter_idx += 1
        hs.append(h)
    if features_adapter is not None:
        assert len(features_adapter) == adapter_idx, 'Wrong features_adapter'

    ########################################### only change here, add camera_condition input ###########################################
    h = self.middle_block(h, emb, context=context, batch_size=b, camera_condition=camera_condition)
    ########################################### only change here, add camera_condition input ###########################################

    for module in self.output_blocks:
        h = torch.cat([h, hs.pop()], dim=1)
        ########################################### only change here, add camera_condition input ###########################################
        h = module(h, emb, context=context, batch_size=b, camera_condition=camera_condition)
        ########################################### only change here, add camera_condition input ###########################################
    h = h.type(x.dtype)
    y = self.out(h)

    # reshape back to (b c t h w)
    y = rearrange(y, '(b t) c h w -> b c t h w', b=b)
    return y


# add camera_condition input to forward of TemporalTransformer
def new_forward_for_TimestepEmbedSequential(self, x, emb, context=None, batch_size=None, camera_condition=None):
    for layer in self:
        if isinstance(layer, TimestepBlock):
            x = layer(x, emb, batch_size=batch_size)
        elif isinstance(layer, SpatialTransformer):
            x = layer(x, context)
        elif isinstance(layer, TemporalTransformer):
            x = rearrange(x, '(b f) c h w -> b c f h w', b=batch_size)
            ########################################### only change here, add camera_condition input ###########################################
            x = layer(x, context, camera_condition=camera_condition)
            ########################################### only change here, add camera_condition input ###########################################
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        else:
            x = layer(x)
    return x


def new_forward_for_TemporalTransformer(self, x, context=None, camera_condition=None):
    b, c, t, h, w = x.shape
    x_in = x
    x = self.norm(x)
    x = rearrange(x, 'b c t h w -> (b h w) c t').contiguous()
    if not self.use_linear:
        x = self.proj_in(x)
    x = rearrange(x, 'bhw c t -> bhw t c').contiguous()
    if self.use_linear:
        x = self.proj_in(x)

    temp_mask = None
    if self.causal_attention:
        # slice the from mask map
        temp_mask = self.mask[:, :t, :t].to(x.device)

    if temp_mask is not None:
        mask = temp_mask.to(x.device)
        mask = repeat(mask, 'l i j -> (l bhw) i j', bhw=b * h * w)
    else:
        mask = None

    if self.only_self_att:
        ## note: if no context is given, cross-attention defaults to self-attention
        for i, block in enumerate(self.transformer_blocks):
            ########################################### only change here, add camera_condition input ###########################################
            x = block(x, mask=mask, camera_condition=camera_condition)
            ########################################### only change here, add camera_condition input ###########################################
        x = rearrange(x, '(b hw) t c -> b hw t c', b=b).contiguous()
    else:
        x = rearrange(x, '(b hw) t c -> b hw t c', b=b).contiguous()
        context = rearrange(context, '(b t) l con -> b t l con', t=t).contiguous()
        for i, block in enumerate(self.transformer_blocks):
            # calculate each batch one by one (since number in shape could not greater then 65,535 for some package)
            for j in range(b):
                context_j = repeat(
                    context[j],
                    't l con -> (t r) l con', r=(h * w) // t, t=t).contiguous()
                ## note: causal mask will not applied in cross-attention case
                x[j] = block(x[j], context=context_j)

    if self.use_linear:
        x = self.proj_out(x)
        x = rearrange(x, 'b (h w) t c -> b c t h w', h=h, w=w).contiguous()
    if not self.use_linear:
        x = rearrange(x, 'b hw t c -> (b hw) c t').contiguous()
        x = self.proj_out(x)
        x = rearrange(x, '(b h w) c t -> b c t h w', b=b, h=h, w=w).contiguous()

    return x + x_in




def new_forward_for_BasicTransformerBlock_of_TemporalTransformer(self, x, context=None, mask=None, camera_condition=None, **kwargs):
    ## implementation tricks: because checkpointing doesn't support non-tensor (e.g. None or scalar) arguments
    forward_method = self._forward
    input_tuple = (x,)  ## should not be (x), otherwise *input_tuple will decouple x into multiple arguments

    if context is not None:
        input_tuple = (x, context)

    if mask is not None:
        forward_method = partial(forward_method, mask=mask)

    if camera_condition is not None:
        forward_method = partial(forward_method, camera_condition=camera_condition)

    return checkpoint(forward_method, input_tuple, self.parameters(), self.checkpoint)

def new__forward_for_BasicTransformerBlock_of_TemporalTransformer(self, x, context=None, mask=None, camera_condition=None):
    x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, mask=mask) + x

    # Add camera pose
    if camera_condition is not None and isinstance(camera_condition, dict) and "RT" in camera_condition:
        RT = camera_condition["RT"]
        B, t, _ = RT.shape  # [B, video_length, pose_dim=12]
        hw = x.shape[0] // B
        RT = RT.reshape(B, t, -1)
        RT = RT.repeat_interleave(repeats=hw, dim=0)                # (bhw, t, 12)
        x = self.cc_projection(torch.cat([x, RT], dim=-1))  # (bhw, t, 12+c) --> linear(c+12, c) --> (bhw, t, c)

    x = self.attn2(self.norm2(x), context=context, mask=mask) + x
    x = self.ff(self.norm3(x)) + x
    return x