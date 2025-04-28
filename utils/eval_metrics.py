import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T
import torch
from torch.utils.data import Dataset
import h5py
import os
import numpy as np
import yaml


def pearson_correlation(targets, predictions, eps=1e-8):
    """
    Unified function to compute Pearson correlation per channel (and timestep if T>1).
    
    Args:
        targets, predictions: Tensors of shape [B, T, C, H, W] where T can be 1 or more
        eps: Small value for numerical stability
        
    Returns:
        - If T=1: Tensor of shape [C] with Pearson r per channel averaged over batch
        - If T>1: Tensor of shape [T, C] with Pearson r per channel and timestep
    """
    B, T, C, H, W = targets.shape
    
    # Reshape to flatten spatial dimensions
    targets = targets.reshape(B, T, C, -1)  # [B, T, C, H*W]
    predictions = predictions.reshape(B, T, C, -1)
    
    # Calculate means and center the data
    target_mean = targets.mean(dim=-1, keepdim=True)
    pred_mean = predictions.mean(dim=-1, keepdim=True)
    
    target_centered = targets - target_mean
    pred_centered = predictions - pred_mean
    
    # Calculate correlation components
    numerator = (target_centered * pred_centered).sum(dim=-1)  # [B, T, C]
    denominator = torch.sqrt(
        (target_centered ** 2).sum(dim=-1) * 
        (pred_centered ** 2).sum(dim=-1) + eps
    )
    
    # Calculate r value per batch item
    r_per_batch = numerator / denominator  # [B, T, C]
    
    # Average over batch dimension
    r_per_channel_time = r_per_batch.mean(dim=0)  # [T, C]
    
    # If T=1, squeeze out the time dimension for simpler output format
    #if T == 1:
    #    r_per_channel_time = r_per_channel_time.squeeze(0)  # [C]
        
    return r_per_channel_time


def nmse(targets, predictions, eps=1e-8):
    """
    Unified function to compute NMSE per channel (and timestep if T>1).
    
    Args:
        targets, predictions: Tensors of shape [B, T, C, H, W] where T can be 1 or more
        eps: Small value for numerical stability
        
    Returns:
        - If T=1: Tensor of shape [C] with NMSE per channel averaged over batch
        - If T>1: Tensor of shape [T, C] with NMSE per channel and timestep
    """
    B, T, C, H, W = targets.shape
    
    # Calculate MSE and normalization per channel
    mse = ((targets - predictions) ** 2).mean(dim=(-2, -1))  # [B, T, C]
    norm = (targets ** 2).mean(dim=(-2, -1))                 # [B, T, C]
    nmse = mse / (norm + eps)
    
    # Average over batch dimension
    nmse_per_channel_time = nmse.mean(dim=0)  # [T, C]
    
    # If T=1, squeeze out the time dimension for simpler output format
    #if T == 1:
    #    nmse_per_channel_time = nmse_per_channel_time.squeeze(0)  # [C]
        
    return nmse_per_channel_time

def vmse_vrmse(targets, predictions, eps=1e-8):
    """
    Compute Variance-Scaled MSE (VMSE) and its square root (VRMSE) per channel and timestep.

    Args:
        targets, predictions: Tensors of shape [B, T, C, H, W]
        eps: Small constant for numerical stability

    Returns:
        vmse: Tensor of shape [T, C]
        vrmse: Tensor of shape [T, C]
    """
    B, T, C, H, W = targets.shape

    # Flatten spatial dimensions
    targets = targets.view(B, T, C, -1)
    predictions = predictions.view(B, T, C, -1)

    # Compute MSE
    mse = ((targets - predictions) ** 2).mean(dim=-1)  # [B, T, C]

    # Compute variance of the truth
    target_mean = targets.mean(dim=-1, keepdim=True)
    target_var = ((targets - target_mean) ** 2).mean(dim=-1)  # [B, T, C]

    # Compute VMSE and VRMSE
    vmse = mse / (target_var + eps)  # [B, T, C]
    vrmse = torch.sqrt(vmse)   # [B, T, C]

    # Average over batch
    vmse = vmse.mean(dim=0)   # [T, C]
    vrmse = vrmse.mean(dim=0) # [T, C]

    return vmse, vrmse
