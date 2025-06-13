import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from layers.attention_layers import FullPatchAttention
from layers.ffn_layers import SwiGLU
from layers.embeddings import PatchEmbedding
from layers.channel_dropout import ChannelDropout

class PatchTransformer(nn.Module):
    """
    Full patch transformer model that processes data with attention along the patch dimension.
    Uses PatchAttention for the attention mechanism.
    
    Args:
        d_model: Feature dimension of each variable's patches
        num_patches: Number of patches/time steps
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    def __init__(self, configs):
        super().__init__()
        
        self.d_input = configs.d_input
        self.d_model = configs.d_model
        self.num_layers = configs.num_layers
        self.num_patches = configs.num_patches
        self.num_heads = configs.num_heads
        self.pred_len = configs.pred_len
        self.patch_length = configs.seq_len // configs.num_patches
        self.dropout = configs.dropout
        self.channel_dropout = configs.channel_dropout
        self.use_norm = configs.use_norm
        
        # Main transformer blocks - one per variable
        self.transformer_blocks = nn.ModuleList([
            nn.ModuleDict({
                'layer_norm': nn.LayerNorm(self.d_model),
                'attn': FullPatchAttention(self.d_model, num_heads=self.num_heads, dropout=self.dropout),
                'ffn': SwiGLU(self.d_model),
                'layer_norm_ffn': nn.LayerNorm(self.d_model),
                'dropout': nn.Dropout(self.dropout)
            }) for _ in range(self.num_layers)
        ])
        
        self.patch = PatchEmbedding(patch_length=self.patch_length, input_dim=self.d_input, embedding_dim=self.d_model)
        self.predictor = nn.Linear(self.d_model * self.num_patches, self.pred_len)
        
        self.channel_dropout = ChannelDropout(p=self.channel_dropout)
        
    def forward(self, x, x_mark=None):
        """
        Args:
            x: Input tensor of shape (B, L, N)
        """
        x = self.channel_dropout(x)
        
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
        
        x = self.patch(x)
        
        b, n, p, f = x.shape
        
        x = rearrange(x, 'b n p f -> b (n p) f')
        
        for block in self.transformer_blocks:
            residual = x
            x = block['layer_norm'](x)
            x = block['attn'](
                q=x,
                k=x,
                v=x
            )
            x = residual + block['dropout'](x)
            
            residual = x
            x = block['layer_norm_ffn'](x)
            x = block['ffn'](x)
            x = residual + block['dropout'](x)
            
        x = rearrange(x, 'b (n p) f -> b n (p f)', n=n, p=p)
        x = self.predictor(x).permute(0, 2, 1)
        
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            x = x + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return x