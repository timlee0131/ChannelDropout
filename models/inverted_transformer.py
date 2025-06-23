import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from layers.attention_layers import InvertedAttention
from layers.ffn_layers import SwiGLU
from channel_dropout import ChannelDropout

class InvertedTransformer(nn.Module):
    """
    Inverted transformer model that processes data with attention along the variable dimension.
    Uses InvertedAttention for the attention mechanism.
    
    Args:
        d_model: Feature dimension of each variable
        num_heads: Number of attention heads
        dropout: Dropout rate
        channel_dropout: Channel dropout rate
    """
    def __init__(self, configs):
        super().__init__()
        
        self.d_input = configs.d_input
        self.d_model = configs.d_model
        self.num_layers = configs.num_layers
        self.num_heads = configs.num_heads
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.dropout = configs.dropout
        self.channel_dropout = configs.channel_dropout
        self.use_norm = configs.use_norm
        
        self.encoder = nn.Linear(self.seq_len, self.d_model)
        
        # Main transformer blocks - one per variable
        self.transformer_blocks = nn.ModuleList([
            nn.ModuleDict({
                'layer_norm': nn.LayerNorm(self.d_model),
                'attn': InvertedAttention(self.d_model, num_heads=self.num_heads, dropout=self.dropout),
                'ffn': SwiGLU(self.d_model),
                'layer_norm_ffn': nn.LayerNorm(self.d_model),
                'dropout': nn.Dropout(self.dropout)
            }) for _ in range(self.num_layers)
        ])
        
        self.predictor = nn.Linear(self.d_model, self.pred_len)
        
        self.channel_dropout = ChannelDropout(p=self.channel_dropout)
        
        # channel dropout scheduling
        self.dropout_scheduler = nn.ModuleList([
            ChannelDropout(p=0.08),
            ChannelDropout(p=0.06),
            ChannelDropout(p=0.04),
            ChannelDropout(p=0.02),
        ])
        
    def forward(self, x, x_mark=None):
        """
        Args:
            x: Input tensor of shape (B, L, N)
        """
        
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
        
        x = x.permute(0, 2, 1)
        
        x = self.encoder(x)
        
        for i, block in enumerate(self.transformer_blocks):
            residual = x
            x = block['layer_norm'](x)
            
            attn_mask = None
            if self.training:
                mask = None
                x, mask = self.dropout_scheduler[i](x)
                
                if mask is not None:
                    mask = mask.squeeze(-1)
                    attn_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
                    diagonal = torch.eye(attn_mask.shape[-1], device=attn_mask.device, dtype=torch.bool)
                    diagonal = diagonal.unsqueeze(0).expand(attn_mask.shape[0], -1, -1)
                    attn_mask = attn_mask | diagonal
                
            x = block['attn'](
                q=x,
                k=x,
                v=x,
                attn_mask=attn_mask,
            )
  
            x = residual + block['dropout'](x)
            
            residual = x
            x = block['layer_norm_ffn'](x)

            # channel dropout for FFN
            # if self.training:
            #     x, _ = self.channel_dropout(x)
            
            x = block['ffn'](x)
            x = residual + block['dropout'](x)
            
        x = self.predictor(x).permute(0, 2, 1)
        
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            x = x + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return x