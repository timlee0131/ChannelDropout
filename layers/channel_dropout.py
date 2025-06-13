import torch
import torch.nn as nn
import math

class ChannelDropout(nn.Module):
    """
    Args:
        p (float): channel dropout rate
        shared_across_batch (bool): If True, same channels dropped for entire batch
        deterministic (bool): If True, always drop the same channels
    
    Shape:
        Input: (B, L, N) 
        Output: (B, L, N) with floor(N * p) channels dropped
    """
    
    def __init__(self, p=0.1, shared_across_batch=False, deterministic=False):
        super(ChannelDropout, self).__init__()
        
        if not 0 <= p < 1:
            raise ValueError(f"dropout probability must be between 0 and 1, got {p}")
        
        self.p = p
        self.shared_across_batch = shared_across_batch
        self.deterministic = deterministic
        
        # For deterministic mode
        self._dropout_indices = None
        self._original_n_channels = None
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, L, N)
        
        Returns:
            Output tensor with floor(N * p) channels dropped
        """
        if not self.training or self.p == 0:
            return x
        
        B, L, N = x.shape
        
        # Calculate exact number of channels to drop
        num_to_drop = int(N * self.p)
        
        if num_to_drop == 0:
            return x  # No channels to drop
        
        if num_to_drop >= N:
            raise ValueError(f"Cannot drop {num_to_drop} channels from {N} total channels")
        
        # Create mask with exactly num_to_drop channels dropped
        if self.shared_across_batch:
            # Same channels dropped for all samples in batch
            mask = self._create_fixed_mask(1, N, num_to_drop, x.device)
            mask = mask.expand(B, N)  # Shape: (B, N)
        else:
            # Different channels can be dropped for each sample
            mask = self._create_fixed_mask(B, N, num_to_drop, x.device)
        
        # Reshape mask to broadcast across time dimension
        mask = mask.unsqueeze(1)  # Shape: (B, 1, N)
        
        # Apply dropout with exact scaling
        # Since we drop exactly num_to_drop channels, scaling is precise
        num_kept = N - num_to_drop
        scale_factor = N / num_kept  # Exact scaling factor
        
        return x * mask.float() * scale_factor
    
    def _create_fixed_mask(self, batch_size, n_channels, num_to_drop, device):
        """Create mask that drops exactly num_to_drop channels"""
        
        if self.deterministic:
            # Use same channels every time for reproducibility
            if (self._dropout_indices is None or 
                self._original_n_channels != n_channels):
                
                torch.manual_seed(42)  # Fixed seed
                self._dropout_indices = torch.randperm(n_channels)[:num_to_drop]
                self._original_n_channels = n_channels
            
            # Create mask for all batches
            mask = torch.ones(batch_size, n_channels, dtype=torch.bool, device=device)
            mask[:, self._dropout_indices] = False
            
        else:
            # Random selection each time
            mask = torch.ones(batch_size, n_channels, dtype=torch.bool, device=device)
            
            for b in range(batch_size):
                # Randomly select exactly num_to_drop channels to drop
                drop_indices = torch.randperm(n_channels, device=device)[:num_to_drop]
                mask[b, drop_indices] = False
        
        return mask
    
    def get_dropout_info(self, n_channels):
        """Get information about dropout for given number of channels"""
        num_to_drop = int(math.floor(n_channels * self.p))
        num_kept = n_channels - num_to_drop
        actual_drop_rate = num_to_drop / n_channels
        scale_factor = n_channels / num_kept if num_kept > 0 else float('inf')
        
        return {
            'channels_total': n_channels,
            'channels_dropped': num_to_drop,
            'channels_kept': num_kept,
            'requested_rate': self.p,
            'actual_rate': actual_drop_rate,
            'scale_factor': scale_factor
        }
    
    def extra_repr(self):
        return (f'p={self.p}, shared_across_batch={self.shared_across_batch}, '
                f'deterministic={self.deterministic}')