import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.model_ae_simple import Model_AE_Simple
from utils.get_data import ShearFlowDataset
import csv


def pearsonr_correlation(targets, predictions, eps=1e-8):
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


def nmse_loss(targets, predictions, eps=1e-8):
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


        # === Config ===
scale = 16
snapshots_per_sample = 5
condition_snapshots = 4
snapshots_to_generate = 1
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Dataset ===
dataset_path = "/home/ldr934/TheWell/the_well-original/the_well/datasets/shear_flow/data"
test_dataset = ShearFlowDataset(data_dir=dataset_path, split="test", snapshots_per_sample=snapshots_per_sample, mode='one-step')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# === Load model ===
ae_model = Model_AE_Simple(state_size=4, in_channels=4, out_channels=4, scale=scale).to(device)
ae_model.eval()

# === Eval loop ===
nmse_all = []
pearson_all = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        observations = batch.to(device)  # [B, 5, 4, 256, 512]
        condition = observations[:, :condition_snapshots]     # [B, 4, C, H, W]
        target = observations[:, condition_snapshots:]        # [B, 1, C, H, W]

        # Predict last frame
        x_in = condition[:, -1]                               # [B, C, H, W]
        # Wrap it to [B, 3, C, H, W] by repeating the input snapshot (or use real conditioning if you want)
        x_in_5d = x_in.unsqueeze(1).repeat(1, 3, 1, 1, 1)  # dummy repeat for test only
        _, x_out_full = ae_model(x_in_5d)
        x_out = x_out_full[:, 0]  # take the "target" snapshot
        prediction = x_out.unsqueeze(1)                       # [B, 1, C, H, W]

        # Denormalize if needed
        prediction = test_dataset.denormalize(prediction.cpu()).to(device)
        target = test_dataset.denormalize(target.cpu()).to(device)

        # Metrics
        nmse = nmse_loss(target, prediction)
        pearson = pearsonr_correlation(target, prediction)
        nmse_all.append(nmse.cpu())
        pearson_all.append(pearson.cpu())

NMSE_per_timestep = torch.stack(nmse_all).mean(dim=0)  # (T, C)
r_per_timestep = torch.stack(pearson_all).mean(dim=0)        # (T, C)

print("NMSE:", NMSE_per_timestep)
print("Pearson:", r_per_timestep)