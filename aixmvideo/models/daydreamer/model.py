# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from einops import rearrange, reduce, repeat
from .utils import get_2d_sincos_pos_embed,get_1d_sincos_pos_embed_from_grid
from .modules import TimestepEmbedder, LabelEmbedder

#################################################################################
#                                       Block                                   #
#################################################################################

def modulate(x, shift, scale, T):
    """
    Args:
        x : (N,S,D)
        shift : (N,D)
        scale : (N,D)
        T : num_frames 16
    Return:
        x : (N,S,D)
    """

    N, M = x.shape[-2], x.shape[-1] # S,D
    B = scale.shape[0]
    x = rearrange(x, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M) # (NT,S,D) -> (N,TS,D)
    x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    x = rearrange(x, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M) 
    return x

class VDTBlock(nn.Module):
    """
    A VDT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, mode='video', num_frames=16, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.num_frames = num_frames
        
        self.mode = mode
        
        ## Temporal Attention Parameters
        if self.mode == 'video':
            
            self.temporal_norm1 = nn.LayerNorm(hidden_size)
            self.temporal_attn = Attention(
              hidden_size, num_heads=num_heads, qkv_bias=True)
            self.temporal_fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, c):
        """
        Args :
            x : (N,S,D)
            c : (N,D)
        Return:
            x : (N,S,D)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1) # (N,6D) -> 6*(N,D)
        T = self.num_frames
        K, N, M = x.shape
        B = K // T
        if self.mode == 'video':

            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T,n=N,m=M)  # (NT,S,D) -> (NS,T,D)
            res_temporal = self.temporal_attn(self.temporal_norm1(x)) # pre layer norm
            res_temporal = rearrange(res_temporal, '(b n) t m -> (b t) n m',b=B,t=T,n=N,m=M)  # (NS,T,D) -> (NT,S,D)
            res_temporal = self.temporal_fc(res_temporal) 
            x = rearrange(x, '(b n) t m -> (b t) n m',b=B,t=T,n=N,m=M) # (NS,T,D) -> (NT,S,D)
            x = x + res_temporal


        attn = self.attn(modulate(self.norm1(x), shift_msa, scale_msa, self.num_frames)) # modulate + spacial-attention
        attn = rearrange(attn, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M)
        attn = gate_msa.unsqueeze(1) * attn
        attn = rearrange(attn, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M)
        x = x + attn

        mlp = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp, self.num_frames))
        mlp = rearrange(mlp, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M)
        mlp = gate_mlp.unsqueeze(1) * mlp
        mlp = rearrange(mlp, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M)
        x = x + mlp


        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class FinalLayer(nn.Module):
    """
    The final layer of VDT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, num_frames):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.num_frames = num_frames

    def forward(self, x, c):
        """
        Args:
            x : (N, T, D)
        Return:
            x : (N, T, patch_size ** 2 * out_channels)
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale, self.num_frames)
        x = self.linear(x)
        return x

    # model = VDT_models[args.model](                      # model = "VDT-L/2"
    #     input_size=latent_size,  # 16
    #     num_classes=args.num_classes, # 1
    #     num_frames=args.num_frames, # 12
    #     mode=args.mode # "video"
    #     # **additional_kwargs

#################################################################################
#                                 Core VDT Model                                #
#################################################################################

class VDT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mode='video',
        num_frames=16
    ):
        super().__init__()
        self.learn_sigma = learn_sigma # True
        self.in_channels = in_channels # 4
        self.out_channels = in_channels * 2 if learn_sigma else in_channels # 8
        self.patch_size = patch_size # 2

        # NCHW -> NMC -> (N,M,D)
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True) # input_size = 32, patch_size = 2, in_channels = 4, hidden_size = 1152
        self.t_embedder = TimestepEmbedder(hidden_size) # (N,D) diffusion timestep
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob) # 
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        self.mode = mode
        if self.mode == 'video':
            self.num_frames = num_frames
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=False)
            self.time_drop = nn.Dropout(p=0)
        else:
            self.num_frames = 1

        self.blocks = nn.ModuleList([
            VDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, mode=mode, num_frames=self.num_frames) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, self.num_frames)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.mode == 'video':
            grid_num_frames = np.arange(self.num_frames, dtype=np.float32)
            time_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], grid_num_frames)
            self.time_embed.data.copy_(torch.from_numpy(time_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in VDT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of VDT.

        Args:
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels

        Return:


        x : (B,T,C,W,H)                              -> reshape
        x : (BT,C,W,H)                               -> x_embedder(patchify) + pos_embed
        x : (BT,S,D) S=patch_num,D=hidden_dim        -> reshape
        x : (BS,T,D)                                 -> +time_embed(1,T,D) 
        x : (BS,T,D)                                 -> reshape
        x : (BT,S,D)                                 -> time_step embedding(N,D) + label_embedding(N,D)
        x : (BT,S,D)  c : (N,D)                      -> VDT_block(x,c)
        x : (BT,S,D)                                 -> FinalLayer
        x : (B,T,patch_size ** 2 * out_channels)     -> reshape
        x : (B,T,C,H,W)
        """
        
        B, T, C, W, H = x.shape 
        x = x.contiguous().view(-1, C, W, H) # （BT) C W H
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        if self.mode == 'video':
            # Temporal embed
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
            x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> (b t) n m',b=B,t=T)
        
        t = self.t_embedder(t)                   # (N, D)  time_embedding = sin_cos_encode + mlp
        y = self.y_embedder(y, self.training)    # (N, D)  label_embedding = nn.embedding

        c = t + y                             # (N, D) condition info

        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)

        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        x = x.view(B, T, x.shape[-3], x.shape[-2], x.shape[-1])
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of VDT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                                   VDT Configs                                 #
#################################################################################

def VDT_L_2(**kwargs):
    return VDT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def VDT_S_2(**kwargs):
    return VDT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


VDT_models = {
    'VDT-L/2':  VDT_L_2,  
    'VDT-S/2':  VDT_S_2, 
}
