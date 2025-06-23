import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union

class MissingDataSimulator(nn.Module):
    """
    Comprehensive missing data simulator for multivariate time series.
    Supports various realistic missing patterns beyond simple channel dropout.
    """
    
    def __init__(self, 
                 missing_rate: float = 0.1,
                 pattern: str = 'random',
                 block_size: int = 4,
                 seed: Optional[int] = None):
        super().__init__()
        self.missing_rate = missing_rate
        self.pattern = pattern
        self.block_size = block_size
        self.seed = seed
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (B, L, N)
        Returns:
            masked_x: Input with missing values masked
            mask: Boolean mask where True=observed, False=missing
        """
        B, L, N = x.shape
        
        if self.pattern == 'random':
            mask = self._random_missing(B, L, N, x.device)
        elif self.pattern == 'channel_block':
            mask = self._channel_block_missing(B, L, N, x.device)
        elif self.pattern == 'temporal_block':
            mask = self._temporal_block_missing(B, L, N, x.device)
        elif self.pattern == 'sensor_failure':
            mask = self._sensor_failure_missing(B, L, N, x.device)
        elif self.pattern == 'intermittent':
            mask = self._intermittent_missing(B, L, N, x.device)
        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")
        
        # Apply masking - use NaN instead of 0 to avoid bias
        masked_x = x.clone()
        masked_x[~mask] = float('nan')
        
        return masked_x, mask
    
    def _random_missing(self, B, L, N, device):
        """Completely random missing values (MCAR)"""
        if self.seed is not None:
            torch.manual_seed(self.seed)
        return torch.rand(B, L, N, device=device) > self.missing_rate
    
    def _channel_block_missing(self, B, L, N, device):
        """Entire channels/variables missing completely"""
        mask = torch.ones(B, L, N, dtype=torch.bool, device=device)
        
        for b in range(B):
            n_missing_channels = int(N * self.missing_rate)
            if n_missing_channels > 0:
                missing_channels = torch.randperm(N)[:n_missing_channels]
                mask[b, :, missing_channels] = False
        
        return mask
    
    def _temporal_block_missing(self, B, L, N, device):
        """Continuous temporal blocks missing (like equipment downtime)"""
        mask = torch.ones(B, L, N, dtype=torch.bool, device=device)
        
        for b in range(B):
            for n in range(N):
                if torch.rand(1) < self.missing_rate:
                    # Create a random block of missing values
                    block_start = torch.randint(0, max(1, L - self.block_size), (1,))
                    block_end = min(L, block_start + self.block_size)
                    mask[b, block_start:block_end, n] = False
        
        return mask
    
    def _sensor_failure_missing(self, B, L, N, device):
        """Simulate sensor failures - channels become missing after certain time"""
        mask = torch.ones(B, L, N, dtype=torch.bool, device=device)
        
        for b in range(B):
            n_failing_sensors = int(N * self.missing_rate)
            if n_failing_sensors > 0:
                failing_sensors = torch.randperm(N)[:n_failing_sensors]
                for sensor in failing_sensors:
                    failure_time = torch.randint(0, L, (1,))
                    mask[b, failure_time:, sensor] = False
        
        return mask
    
    def _intermittent_missing(self, B, L, N, device):
        """Intermittent missing patterns - comes and goes"""
        mask = torch.ones(B, L, N, dtype=torch.bool, device=device)
        
        for b in range(B):
            for n in range(N):
                if torch.rand(1) < self.missing_rate:
                    # Create multiple small gaps
                    n_gaps = torch.randint(1, 4, (1,))
                    for _ in range(n_gaps):
                        gap_start = torch.randint(0, L, (1,))
                        gap_size = torch.randint(1, min(5, L - gap_start + 1), (1,))
                        gap_end = min(L, gap_start + gap_size)
                        mask[b, gap_start:gap_end, n] = False
        
        return mask


class MissingDataHandler(nn.Module):
    """
    Handles missing data during model forward pass.
    Multiple strategies for dealing with NaN values.
    """
    
    def __init__(self, strategy: str = 'mean_fill', fill_value: float = 0.0):
        super().__init__()
        self.strategy = strategy
        self.fill_value = fill_value
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input with potential NaN values (B, L, N)
            mask: Optional mask (True=observed, False=missing)
        """
        if mask is None:
            # Create mask from NaN values
            mask = ~torch.isnan(x)
        
        if self.strategy == 'zero_fill':
            x_filled = torch.where(torch.isnan(x), torch.tensor(0.0, device=x.device), x)
            return x_filled, mask
            
        elif self.strategy == 'forward_fill':
            x_filled = self._forward_fill(x, mask)
            return x_filled, mask
            
        elif self.strategy == 'mean_fill':
            x_filled = self._mean_fill(x, mask)
            return x_filled, mask
            
        elif self.strategy == 'mask_attention':
            # Keep NaN for attention masking
            return x, mask
            
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _forward_fill(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward fill missing values"""
        x_filled = x.clone()
        B, L, N = x.shape
        
        for b in range(B):
            for n in range(N):
                last_valid = None
                for l in range(L):
                    if mask[b, l, n]:
                        last_valid = x[b, l, n]
                    elif last_valid is not None:
                        x_filled[b, l, n] = last_valid
                    else:
                        x_filled[b, l, n] = 0.0  # No previous value available
        
        return x_filled
    
    def _mean_fill(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Fill with channel-wise mean of observed values"""
        x_filled = x.clone()
        
        for n in range(x.shape[-1]):
            channel_data = x[:, :, n]
            channel_mask = mask[:, :, n]
            
            if channel_mask.sum() > 0:  # Has some observed values
                channel_mean = channel_data[channel_mask].mean()
                x_filled[:, :, n] = torch.where(
                    torch.isnan(channel_data), 
                    channel_mean, 
                    channel_data
                )
            else:
                # No observed values in this channel
                x_filled[:, :, n] = 0.0
        
        return x_filled