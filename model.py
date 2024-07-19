import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional
import iqk_cpp


# Optimized matrix multiplication function with optional row mapping
def iqk_mul_mat(A: torch.Tensor, B: torch.Tensor, row_mapping: Optional[torch.Tensor] = None) -> torch.Tensor:
    A = A.detach().cpu().contiguous()
    B = B.detach().cpu().contiguous()
    
    Nx, Ny = A.shape
    Ne00, Ne01 = B.shape
    Ne0 = Ne01
    
    if A.dtype == torch.quint4x2:
        typeA = 1
    elif A.dtype == torch.quint8:
        typeA = 2
    else:
        raise ValueError(f"Unsupported dtype for A: {A.dtype}")
    
    if B.dtype != torch.float32:
        B = B.to(torch.float32)
    
    if row_mapping is not None:
        row_mapping = row_mapping.detach().cpu().contiguous()
        if row_mapping.dtype != torch.int32:
            row_mapping = row_mapping.to(torch.int32)
        nb1 = B.stride(0)
        nb2 = B.stride(1) if B.dim() > 1 else 0
    else:
        nb1 = Ne0 * 4
        nb2 = 0
    
    C = torch.empty((Nx, Ne0), dtype=torch.float32)
    iqk_cpp.mul_mat(Nx, Ny, Ne0, Ne00, typeA, A.data_ptr(), B.data_ptr(), C.data_ptr(), nb1, nb2, row_mapping.data_ptr() if row_mapping is not None else None)
    
    return C


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class FlashAttentionLayerNorm(nn.Module):
    def __init__(self, d_model, nhead):
        super(FlashAttentionLayerNorm, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, _ = x.shape
        x_norm = self.layer_norm(x)
        qkv = self.qkv(x_norm).reshape(B, L, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = iqk_mul_mat(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = iqk_mul_mat(attn, v).transpose(1, 2).reshape(B, L, self.d_model)
        return out


class RMSNLayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-8):
        super(RMSNLayerNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features))
        self.shift = nn.Parameter(torch.zeros(num_features))
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        x = self.layer_norm(x)
        x = x * self.scale + self.shift
        return x


class OptimizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(OptimizedLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return iqk_mul_mat(x, self.weight.t()) + self.bias


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., dropout=0.1):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.attn = FlashAttentionLayerNorm(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.rmsn1 = RMSNLayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            OptimizedLinear(int(dim * mlp_ratio), dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.rmsn2 = RMSNLayerNorm(dim)
        self.drop_path = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        attn_windows = self.window_partition(x)
        attn_windows = attn_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(attn_windows)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = self.window_reverse(attn_windows, H, W)
        
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W, C)
        x = self.rmsn1(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.rmsn2(x)
        return x

    def window_partition(self, x):
        B, H, W, C = x.shape
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C)
        return windows

    def window_reverse(self, windows, H, W):
        B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
        x = windows.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x


class SwinTransformer(nn.Module):
    def __init__(self, config):
        super(SwinTransformer, self).__init__()
        self.num_layers = len(config['depths'])
        self.embed_dim = config['embed_dim']

        self.patch_embed = nn.Conv2d(config['in_chans'], self.embed_dim, kernel_size=config['patch_size'], stride=config['patch_size'])
        self.pos_drop = nn.Dropout(p=config['dropout_rate'])

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList([
                SwinTransformerBlock(
                    dim=int(self.embed_dim * 2 ** i_layer),
                    num_heads=config['num_heads'][i_layer],
                    window_size=config['window_size'],
                    shift_size=0 if (j % 2 == 0) else config['window_size'] // 2,
                    mlp_ratio=config['mlp_ratio'],
                    dropout=config['dropout_rate']
                ) for j in range(config['depths'][i_layer])
            ])
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.embed_dim * 2 ** (self.num_layers - 1))
        self.head = nn.Linear(self.embed_dim * 2 ** (self.num_layers - 1), config['num_classes'])

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        for layer in self.layers:
            for block in layer:
                x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)
