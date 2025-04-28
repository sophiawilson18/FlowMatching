import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


from model.model_ae import Model_AE
from utils.get_data import ShearFlowDataset

from argparse import ArgumentParser, Namespace as ArgsNamespace
from einops import rearrange


np.random.seed(123)
os.environ['PYTHONHASHSEED'] = str(123)
torch.manual_seed(123)
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = device = torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)
torch.backends.cudnn.deterministic = True    
torch.backends.cudnn.benchmark = False


print("Loading data")
in_channels = 4
out_channels = 4
enc_mid_channels = 96
dec_mid_channels = 192
state_res = [32,64]
state_size = 8
data_dir = "/home/ldr934/TheWell/the_well-original/the_well/datasets/shear_flow/data"
val_dataset = ShearFlowDataset(data_dir=data_dir, split="valid", snapshots_per_sample=3)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

print("Loading model")
AE_PATH = "ae/checkpoints/AE_final.pt"
ae_model = Model_AE(state_size=state_size, in_channels=in_channels, out_channels=out_channels, enc_mid_channels=enc_mid_channels, dec_mid_channels=dec_mid_channels)
ae_model.load_state_dict(torch.load(AE_PATH, map_location=device))
ae_model.to(device)
ae_model.eval()

print("Passing data to encoder")
N = 0
for batch in val_loader:
    if N >= 1:  
        break
    N += 1
    # Fetch data
    observations = batch.to(device)

###### Step 2: Encode snapshots with pretrained encoder
# Encoder
# input snapshots: [b, 3, num_channels, height, width]
with torch.no_grad():
    ae_model.encoder.eval()
    input_snapshots = rearrange(observations, "b n c h w -> (b n) c h w")  # shape: [B * 3, C, H, W]
    latents = ae_model.encoder(input_snapshots)


print("Plotting")
# Assume latents is [3, 8, 32, 64]
sample_idx = 0
channels = latents[sample_idx]  # shape: [8, 32, 64]

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()

for i in range(8):
    axes[i].imshow(channels[i].cpu(), cmap='viridis')
    axes[i].set_title(f'Channel {i}')
    axes[i].axis('off')

plt.suptitle("Latent Channels for Sample 0")
plt.tight_layout()
plt.show()

# save plot
plt.savefig("AE_latens.pdf")
print("Plot saved")