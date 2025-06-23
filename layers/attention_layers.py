import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class FullPatchAttention(nn.Module):
    def __init__(self, d_model, num_heads=1, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.q = nn.Linear(d_model, d_model * num_heads)
        self.k = nn.Linear(d_model, d_model * num_heads)
        self.v = nn.Linear(d_model, d_model * num_heads)
        
        self.out = nn.Linear(d_model * num_heads, d_model)
    
    def forward(self, q, k, v, is_causal=False):        
        # Project queries, keys, and values
        q = self.q(q)  # (batch_size, n_vars, d_model * num_heads)
        k = self.k(k)  # (batch_size, n_vars, d_model * num_heads)
        v = self.v(v)  # (batch_size, n_vars, d_model * num_heads)
        
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, is_causal=is_causal)
        
        attn = rearrange(attn, 'b h n d -> b n (h d)')
        
        return self.out(attn)

class InvertedAttention(nn.Module):
    def __init__(self, d_model, num_heads=1, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.q = nn.Linear(d_model, d_model * num_heads)
        self.k = nn.Linear(d_model, d_model * num_heads)
        self.v = nn.Linear(d_model, d_model * num_heads)
        
        self.out = nn.Linear(d_model * num_heads, d_model)
    
    def forward(self, q, k, v, attn_mask=None, is_causal=False):        
        # Project queries, keys, and values
        q = self.q(q)  # (batch_size, n_vars, d_model * num_heads)
        k = self.k(k)  # (batch_size, n_vars, d_model * num_heads)
        v = self.v(v)  # (batch_size, n_vars, d_model * num_heads)
        
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, is_causal=is_causal, attn_mask=attn_mask)
        
        attn = rearrange(attn, 'b h n d -> b n (h d)')
        
        return self.out(attn)

class ACIAttention(nn.Module):
    def __init__(self, d_model, num_heads=1, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.q = nn.Linear(d_model, d_model * num_heads)
        self.k = nn.Linear(d_model, d_model * num_heads)
        self.v = nn.Linear(d_model, d_model * num_heads)
        
        self.out = nn.Linear(d_model * num_heads, d_model)
    
    def forward(self, q, k, v, attn_mask=None, is_causal=False):        
        # Project queries, keys, and values
        q = self.q(q)  # (batch_size, n_vars, d_model * num_heads)
        k = self.k(k)  # (batch_size, n_vars, d_model * num_heads)
        v = self.v(v)  # (batch_size, n_vars, d_model * num_heads)
        
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, is_causal=is_causal, attn_mask=attn_mask)
        
        attn = rearrange(attn, 'b h n d -> b n (h d)')
        
        return self.out(attn)