"""
This script performs a rollout using a trained Flow Matching (FM) model on a specified dataset.
"""
# ---------------- Imports ----------------
import os
import csv
from copy import deepcopy
from argparse import ArgumentParser, Namespace as ArgsNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from einops import rearrange
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from model.fm import FM_model
from model.ae_fixed import AE_fixed
from model.ae_trainable import AE_trainable
from model.endtoend_ae_trainable import Model_EndtoEnd 
from utils.get_data import SimpleFlowDataset, EvalSimpleFlowDataset, ShearFlowDataset
from utils.plots import add_field_plot


# ---------------- Argument Parsing ----------------
def parse_args() -> ArgsNamespace:
    parser = ArgumentParser()
    # Required
    parser.add_argument("--run-name", type=str, required=True, help="Name of the current run.")
    
    # Dataset and model
    parser.add_argument("--dataset", type=str, default='shearflow', help="Name of dataset.")
    parser.add_argument("--snapshots-per-sample", type=int, default=25, help="Number of snapshots per sample.")
    parser.add_argument("--condition-snapshots", type=int, default=5, help="Number of conditioning snapshots.")
    parser.add_argument("--snapshots-to-generate", type=int, default=20, help="Number of rollout snapshots.")
    parser.add_argument("--test_batch_size", type=int, default=16, help="Test batch size.")
    
    # Sampling
    parser.add_argument("--sampling_steps", type=int, default=10, help="Number of integration steps.")    
    parser.add_argument("--solver", type=str, default='euler', help="Sampler scheme.")
    parser.add_argument("--probpath_option", type=str, default='ours', help="Probability path strategy.")

    # Model config
    parser.add_argument("--sigma", type=float, default=0.01, help="Sigma for FM.")
    parser.add_argument("--sigma_min", type=float, default=0.001, help="Minimum sigma.")
    parser.add_argument("--sigma_sam", type=float, default=0.0, help="Sampling sigma.")

    # Checkpoint paths
    parser.add_argument("--path_to_checkpoints", type=str, default='/home/ldr934/FlowMatching/checkpoints/', help="Checkpoint path.")
    parser.add_argument("--path_to_results", type=str, default='/home/ldr934/FlowMatching/results/', help="Path to save the test results.")
    parser.add_argument("--train_option", type=str, default='data-space', help="Training option: end-to-end, data-space, or separate.")
    parser.add_argument("--scale", type=int, default=16, help="Scale for data-space training.")

    # Misc
    parser.add_argument("--random-seed", type=int, default=1543, help="Random seed.")

    return parser.parse_args()

args = parse_args()


# ---------------- Device and Seed Setup ----------------
print('Launching processes...')
def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    """
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.random_seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- Data Preparation ----------------
if args.dataset == 'shearflow':
    in_channels = 4
    out_channels = 4

    if args.train_option == "end-to-end" or args.train_option == "separate":
        enc_mid_channels = 96
        dec_mid_channels = 192
        state_res = [32,64]
        state_size = 8

    elif args.train_option == "data-space":
        state_res = [int(256/args.scale), int(512/args.scale)]
        state_size = 4
        scale = args.scale

    data_dir = "/home/ldr934/TheWell/the_well-original/the_well/datasets/shear_flow/data"
    test_dataset = ShearFlowDataset(data_dir=data_dir, split="test", snapshots_per_sample=args.snapshots_per_sample, mode="rollout")
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)


# ---------------- Model Loading ----------------
print("Loading AE model...") 
if args.train_option == "end-to-end" or args.train_option == "separate":
    AE_PATH = args.path_to_checkpoints + "ae_" + args.dataset + "_best_data_subset.pt"
    ae_model = AE_trainable(state_size=state_size, in_channels=in_channels, out_channels=out_channels, enc_mid_channels=enc_mid_channels, dec_mid_channels=dec_mid_channels)
    ae_model.load_state_dict(torch.load(AE_PATH, map_location=device))
    ae_model.to(device)
    ae_model.eval()

elif args.train_option == "data-space":
    ae_model = AE_fixed(state_size=state_size, in_channels=in_channels, out_channels=out_channels, scale=scale)
    ae_model.to(device)
    ae_model.eval()
    

print("Loading FM model...")
FM_PATH = args.path_to_checkpoints + args.run_name + args.dataset + ".pt" 

model = FM_model(
    ae_model.encoder, ae_model.decoder,
    state_size=state_size, state_res=state_res,
    ours_sigma=args.sigma, sigma_min=args.sigma_min, sigma_sam=args.sigma_sam
)
model.load_state_dict(torch.load(FM_PATH, map_location=device))
model.to(device)
model.eval()

# ---------------- Rollout ----------------
with torch.no_grad():
    test_gen = tqdm(test_loader, desc="Rollout")

    for i, batch in enumerate(test_gen):
        if i > 10:
            break
        observations = batch.to(device)
        condition_snapshots = observations[:, :args.condition_snapshots]
        targets = observations[:,args.condition_snapshots:]

        generated_observations = model.generate_snapshots(
            observations=condition_snapshots,
            num_condition_snapshots=args.condition_snapshots, 
            num_snapshots=args.snapshots_to_generate, 
            steps=args.sampling_steps, 
            option=args.probpath_option)
        
        predictions = generated_observations[:,args.condition_snapshots:] # [B, T, C, H, W]
    
        # Plotting
        nrow = 8
        ncol = args.condition_snapshots + args.snapshots_to_generate

        obs = test_dataset.denormalize(observations[0]).unsqueeze(0).cpu().numpy()  # (1, T, C, H, W)
        gen = test_dataset.denormalize(generated_observations[0]).unsqueeze(0).cpu().numpy()

        obs[:, :, [0, 1], :, :] = obs[:, :, [1, 0], :, :] 
        gen[:, :, [0, 1], :, :] = gen[:, :, [1, 0], :, :]

        f, axarr = plt.subplots(nrow, ncol, figsize=(25, 10))  
        plt.subplots_adjust(left=0.05, right=0.86, wspace=0.02, hspace=0.2)

        # Plot all 4 fields
        add_field_plot(obs, gen, 0, 1, channel=0, label="tracer")      # Tracer
        add_field_plot(obs, gen, 2, 3, channel=1, label="pressure")    # Pressure
        add_field_plot(obs, gen, 4, 5, channel=2, label="x-velocity")  # x-velocity
        add_field_plot(obs, gen, 6, 7, channel=3, label="y-velocity")  # y-velocity

        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space on right for colorbars  

        directory = os.path.join(args.path_to_results, args.run_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        savename = f'/rollout_{args.snapshots_to_generate}_batch{i}.pdf'
        plt.savefig(directory + savename)
        plt.close()
        print(f"Saved plot to {directory + savename}")

        obs_np = obs.astype(np.float32)
        gen_np = gen.astype(np.float32)

        np.savez_compressed(f"{directory}/rollout.npz", y_pred=gen, y_ref=obs)
        print(f"Saved rollout results to {directory}/rollout.npz")
