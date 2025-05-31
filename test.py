import os
import os.path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from model.model_endtoend import Model_EndtoEnd 
from model.model import Model
from model.model_ae import Model_AE
from model.model_ae_simple import Model_AE_Simple
from utils.get_data import ShearFlowDataset
from utils.eval_metrics import pearson_correlation, nmse, vmse_vrmse, pearson_correlation_unified, vmse_vrmse_unified

from argparse import ArgumentParser, Namespace as ArgsNamespace
import csv


        

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
    parser.add_argument("--path_to_checkpoints", type=str, default='/home/ldr934/minFlowMatching/checkpoints/', help="Checkpoint path.")
    parser.add_argument("--path_to_results", type=str, default='/home/ldr934/minFlowMatching/results/', help="Path to save the test results.")

    # Misc
    parser.add_argument("--random-seed", type=int, default=1543, help="Random seed.")

    parser.add_argument("--train_option", type=str, default='data-space', help="Training option: end-to-end, data-space, or separate.")
    parser.add_argument("--scale", type=int, default=16, help="Scale for data-space training.")

    return parser.parse_args()

args = parse_args()


# Launch processes.
print('Launching processes...')

# Initialize
np.random.seed(args.random_seed)
os.environ['PYTHONHASHSEED'] = str(args.random_seed)
torch.manual_seed(args.random_seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.random_seed)
torch.backends.cudnn.deterministic = True    
torch.backends.cudnn.benchmark = False

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
    if args.snapshots_to_generate == 1:
        test_dataset = ShearFlowDataset(data_dir=data_dir, split="test", snapshots_per_sample=args.snapshots_per_sample, mode='one-step')          # valid set is used for testing !!!!
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    else:
        test_dataset = ShearFlowDataset(data_dir=data_dir, split="test", snapshots_per_sample=args.snapshots_per_sample, mode="rollout")
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)



# Load model
print("Loading AE model...") 
if args.train_option == "end-to-end" or args.train_option == "separate":
    AE_PATH = args.path_to_checkpoints + "ae_" + args.dataset + "_best_data_subset.pt"
    ae_model = Model_AE(state_size=state_size, in_channels=in_channels, out_channels=out_channels, enc_mid_channels=enc_mid_channels, dec_mid_channels=dec_mid_channels)
    ae_model.load_state_dict(torch.load(AE_PATH, map_location=device))
    ae_model.to(device)
    ae_model.eval()

elif args.train_option == "data-space":
    ae_model = Model_AE_Simple(state_size=state_size, in_channels=in_channels, out_channels=out_channels, scale=scale)
    ae_model.to(device)
    ae_model.eval()

print("Loading FM model...")
FM_PATH = args.path_to_checkpoints + args.run_name + "_FM_best.pt"



model = Model(
    ae_model.encoder, ae_model.decoder,
    state_size=state_size, state_res=state_res,
    ours_sigma=args.sigma, sigma_min=args.sigma_min, sigma_sam=args.sigma_sam
)
model.load_state_dict(torch.load(FM_PATH, map_location=device))
model.to(device)
model.eval()

# Testing
nmse_all = []
r_all = []
vmse = []
vrmse = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        observations = batch.cuda()
        condition_snapshots = observations[:, :args.condition_snapshots]
        targets = observations[:, args.condition_snapshots:]

        generated_observations = model.generate_snapshots(
            observations=condition_snapshots,
            num_condition_snapshots=args.condition_snapshots,
            num_snapshots=args.snapshots_to_generate,
            steps=args.sampling_steps,
            option=args.probpath_option
        )

        predictions = generated_observations[:, args.condition_snapshots:]

        targets_denom = test_dataset.denormalize(targets.detach().cpu()).to(device)
        predictions_denom = test_dataset.denormalize(predictions.detach().cpu()).to(device)

        # Use the unified functions for metrics calculation
        # Note: Make sure the dimensions are correct - we're expecting [B, T, C, H, W]
        nmse_ = nmse(targets_denom, predictions_denom)
        r_ = pearson_correlation_unified(targets_denom, predictions_denom)
        vmse_, vrmse_ = vmse_vrmse_unified(targets_denom, predictions_denom)

        nmse_all.append(nmse_.cpu())
        r_all.append(r_.cpu())
        vmse.append(vmse_.cpu())
        vrmse.append(vrmse_.cpu())


nmse_all = torch.stack(nmse_all).mean(dim=0)  # (T, C)
r_all = torch.stack(r_all).mean(dim=0)        # (T, C)
vmse = torch.stack(vmse).mean(dim=0)        # (T, C)
vrmse = torch.stack(vrmse).mean(dim=0)        # (T, C)

# Print metrics
print("NMSE:", nmse_all)
print("Pearson r:", r_all)
print("VMSE:", vmse)
print("VRMSE:", vrmse)


# Save results
results_dir = os.path.join(args.path_to_results, args.run_name)
os.makedirs(results_dir, exist_ok=True)
csv_path = os.path.join(results_dir, 'test_loss.csv')

if args.snapshots_to_generate == 1:
    file_name_nmse = f"fm_nmse_one_step.csv"
    file_name_pearson = f"fm_pearson_one_step.csv"
    file_name_vmse = f"fm_vmse_one_step.csv"
    file_name_vrmse = f"fm_vrmse_one_step.csv"
else:
    file_name_nmse = f"fm_nmse_rollout.csv"
    file_name_pearson = f"fm_pearson_rollout.csv"
    file_name_vmse = f"fm_vmse_rollout.csv"
    file_name_vrmse = f"fm_vrmse_rollout.csv"
    


# Swap fields 0 and 1 before saving NMSE
with open(osp.join(results_dir, file_name_nmse), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestep"] + [f"Field_{i}" for i in range(nmse_all.shape[1])])
    for t in range(nmse_all.shape[0]):
        row = nmse_all[t].tolist()
        row[0], row[1] = row[1], row[0]  # swap field 0 and 1
        writer.writerow([t] + row)

# Swap fields 0 and 1 before saving Pearson r
with open(osp.join(results_dir, file_name_pearson), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestep"] + [f"Field_{i}" for i in range(r_all.shape[1])])
    for t in range(r_all.shape[0]):
        row = r_all[t].tolist()
        row[0], row[1] = row[1], row[0]  # swap field 0 and 1
        writer.writerow([t] + row)

# Save VMSE
with open(osp.join(results_dir, file_name_vmse), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestep"] + [f"Field_{i}" for i in range(vmse.shape[1])])
    for t in range(vmse.shape[0]):
        row = vmse[t].tolist()
        row[0], row[1] = row[1], row[0]  # swap field 0 and 1
        writer.writerow([t] + row)

# Save VRMSE
with open(osp.join(results_dir, file_name_vrmse), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestep"] + [f"Field_{i}" for i in range(vrmse.shape[1])])
    for t in range(vrmse.shape[0]):
        row = vrmse[t].tolist()
        row[0], row[1] = row[1], row[0]  # swap field 0 and 1
        writer.writerow([t] + row)
print(f"Results saved to {results_dir}")
