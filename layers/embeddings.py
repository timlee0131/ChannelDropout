import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    """
    Time series patching and embedding as done in PatchTST.
    """
    def __init__(self, patch_length, input_dim, embedding_dim, max_seq_len=5000, dropout=0.1):
        super().__init__()
        
        self.patch_length = patch_length
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # A single, learned projection that gets applied to all patches
        self.projection = nn.Linear(patch_length * input_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer('temporal_pe', self._create_sinusoidal_encoding(max_seq_len, input_dim))
    
    def _create_sinusoidal_encoding(self, max_seq_length, d_model):
        """
        Creates sinusoidal positional encoding matrix
        """
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe  # Shape: (max_seq_length, d_model)
    
    def _add_periodic_encodings(self, x, periods=[24, 7*24, 365*24]):
        """Add explicit encodings for known periodicities"""
        batch_size, num_vars, seq_length, input_dim = x.shape
        
        for period in periods:
            # Create cyclic features
            cycle_sin = torch.sin(2 * math.pi * torch.arange(seq_length) / period).to(x.device)
            cycle_cos = torch.cos(2 * math.pi * torch.arange(seq_length) / period).to(x.device)
            
            # Add to input
            cycle_encoding = torch.stack([cycle_sin, cycle_cos], dim=-1)
            x = torch.cat([x, cycle_encoding.unsqueeze(0).unsqueeze(1).expand(batch_size, num_vars, -1, -1)], dim=-1)
        
        return x
    
    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, num_variables, seq_length, input_dim)
        Returns: Tensor of shape (batch_size, num_variables, num_patches, embedding_dim)
        """
        if x.ndim == 3: # if incoming data is of shape (batch_size B, seq_length L, input_dim N)
            x = x.permute(0, 2, 1).unsqueeze(-1)
        
        batch_size, num_vars, seq_length, input_dim = x.shape
        
        t_pe = self.temporal_pe[:seq_length, :]
        x = x + t_pe.unsqueeze(0).unsqueeze(1)
        
        # # period encoding ended up not being effective against traffic -- could be useful elsewhere
        # x = self._add_periodic_encodings(x)

        # Ensure sequence length is divisible by patch length
        if seq_length % self.patch_length != 0:
            raise ValueError(f"Sequence length ({seq_length}) must be divisible by patch length ({self.patch_length})")
        
        num_patches = seq_length // self.patch_length
        
        # Reshape into patches: (batch, num_vars, num_patches, patch_length, input_dim)
        patches = x.reshape(batch_size, num_vars, num_patches, self.patch_length, input_dim)
        
        # Reshape for projection: (batch * num_vars* num_patches, patch_length * input_dim)
        flattened = patches.reshape(-1, self.patch_length * input_dim)
        
        # Project to embedding dimension
        embeddings = self.projection(flattened)
        
        embeddings = self.dropout(embeddings) if self.training else embeddings
        
        # Reshape back to (batch, num_patches, num_vars, embedding_dim)
        embeddings = embeddings.reshape(batch_size, num_vars, num_patches, self.embedding_dim)
        
        # returns (batch, num_vars, num_patches, embedding_dim) or (B, N, P, D)
        return embeddings