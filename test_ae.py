import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.model_ae_simple import Model_AE_Simple
from utils.get_data import ShearFlowDataset
from argparse import ArgumentParser, Namespace as ArgsNamespace
import matplotlib.pyplot as plt

def parse_args() -> ArgsNamespace:
    parser = ArgumentParser()
    parser.add_argument("--snapshots-per-sample", type=int, default=5, help="Number of snapshots per sample.")
    parser.add_argument("--condition-snapshots", type=int, default=4, help="Number of snapshots per sample.")
    parser.add_argument("--snapshots-to-generate", type=int, default=1, help="Number of snapshots per sample.") 
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Train batch size.")
    parser.add_argument("--scale", type=int, default=4, help="Scale for the encoder and decoder.")
    
    return parser.parse_args()

args = parse_args()

os.environ['PYTHONHASHSEED'] = str(args.random_seed)
torch.manual_seed(args.random_seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

in_channels = 4
out_channels = 4

state_res = [int(256/args.scale), int(512/args.scale)]
state_size = 4
scale = args.scale

data_dir = "/home/ldr934/TheWell/the_well-original/the_well/datasets/shear_flow/data"
train_dataset = ShearFlowDataset(data_dir=data_dir, split="train", snapshots_per_sample=args.snapshots_per_sample)
train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True) 

ae_model = Model_AE_Simple(state_size=state_size, in_channels=in_channels, out_channels=out_channels, scale=scale)
ae_model.to(device)
ae_model.eval()

loss_fn = nn.MSELoss()
losses = []

with torch.no_grad():
    for batch_idx, batch in enumerate(train_loader):
        observations = batch.to(device)
        input_snapshots, reconstructed_snapshots = ae_model(observations)
        loss = loss_fn(input_snapshots, reconstructed_snapshots)
        losses.append(loss.item())

        if batch_idx < 0:  # Visualize first 5 batches
            print(f"Batch {batch_idx}, Loss: {loss.item()}")
            # Visualize first example in batch
            input_example = input_snapshots[0].cpu().numpy()        # shape [3, C, H, W]
            recon_example = reconstructed_snapshots[0].cpu().numpy()

            n_snapshots = input_example.shape[0]
            fig, axes = plt.subplots(2, n_snapshots, figsize=(4 * n_snapshots, 6))

            titles = ["(target)", "(reference)", "(conditioning)"]
            for i in range(n_snapshots):
                im_in = input_example[i][0]  # take first channel
                im_rec = recon_example[i][0]

                axes[0, i].imshow(im_in, cmap='RdBu')
                axes[0, i].set_title(f"Input {titles[i]}")
                axes[0, i].axis("off")

                axes[1, i].imshow(im_rec, cmap='RdBu')
                axes[1, i].set_title(f"Reconstruction {titles[i]}")
                axes[1, i].axis("off")

            plt.suptitle(f"Reconstruction loss (MSE): {loss.item()}")

            plt.tight_layout()
            plt.show()
            plt.savefig(f"ae_reconstructions_scale{args.scale}/ae_reconstruction_batch_{batch_idx}.png")
            plt.close()
              # only visualize one batch
        if batch_idx > 100:
            break


losss = np.array(losses)
mean_loss = np.mean(losss)
std_loss = np.std(losss)
print(f"Mean loss: {mean_loss}, Std loss: {std_loss}")
