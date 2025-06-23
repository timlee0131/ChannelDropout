import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class GumbelSoftmaxChannelMask(nn.Module):
    """
    Learns which channels should be isolated (self-attention only) vs cross-channel interactions.
    Uses Gumbel Softmax for differentiable discrete decisions.
    """
    def __init__(self, temperature=1.0, sharp=True):
        super().__init__()
        self.temperature = temperature
        self.sharp = sharp
        
        self.channel_logits = None
        
    def forward(self, num_channels, training=True, device=None):
        """
        Returns:
            mask: Boolean mask (n_channels, n_channels) where True = allow attention
                  False = block cross-channel attention (force self-attention)
        """
        if self.channel_logits is None:
            self.channel_logits = nn.Parameter(torch.zeros(num_channels))
        
        # Sample binary decisions for each channel (isolated vs multivariate)
        if training:
            # Gumbel Softmax for differentiable sampling
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.channel_logits) + 1e-8) + 1e-8)
            logits = (self.channel_logits + gumbel_noise) / self.temperature
            isolated_probs = torch.sigmoid(logits)
            
            if self.sharp:
                # Straight-through estimator
                isolated_binary = (isolated_probs > 0.5).float()
                isolated_binary = isolated_binary + isolated_probs - isolated_probs.detach()
            else:
                isolated_binary = isolated_probs
        else:
            # Deterministic at test time
            isolated_binary = (torch.sigmoid(self.channel_logits) > 0.5).float()
        
        # Create attention mask
        # If channel i is isolated, it can only attend to itself
        mask = torch.eye(num_channels, device=self.channel_logits.device)
        
        # Add cross-channel interactions for non-isolated channels
        multivariate_mask = (1 - isolated_binary).unsqueeze(0) * (1 - isolated_binary).unsqueeze(1)
        mask = mask + multivariate_mask
        
        return mask.bool().to(device), isolated_binary.to(device)