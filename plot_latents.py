import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
import time

from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from model.model_endtoend import Model_EndtoEnd 
from model.model import Model
from model.model_ae import Model_AE
from model.model_ae_simple import Model_AE_Simple
from model.model import metric
from utils.get_data import SimpleFlowDataset, EvalSimpleFlowDataset, ShearFlowDataset

import torch.optim.lr_scheduler as lr_scheduler
from einops import rearrange
from transformers import get_cosine_schedule_with_warmup

from transformers import get_polynomial_decay_schedule_with_warmup
from argparse import ArgumentParser, Namespace as ArgsNamespace
import csv
from mpl_toolkits.axes_grid1 import make_axes_locatable

def add_field_plot(obs, gen, row_gt, row_pred, channel, label, args, cmap='RdBu'): # (12, 5, 4, 256, 512)
    global f, axarr, ncol  # ensure these are accessible

    # Compute symmetric color range
    data_gt = obs[0, :, channel]
    data_pred = gen[0, :, channel]
    vmax = np.max(np.abs(np.concatenate([data_gt, data_pred])))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    for i in range(ncol):
        axarr[row_gt, i].imshow(obs[0, i, channel], cmap=cmap, norm=norm) # plot first batch, loops over all snapshots
        axarr[row_pred, i].imshow(gen[0, i, channel], cmap=cmap, norm=norm)

        for ax in [axarr[row_gt, i], axarr[row_pred, i]]:
            ax.set_xticks([])
            ax.set_yticks([])

        # Only title on the top row
        if row_gt == 0:
            axarr[row_gt, i].set_title("Cond." if i < args.condition_snapshots else "Rollout", fontsize=12)

    # Set single ylabel on the far-left of each row pair
    axarr[row_gt, 0].set_ylabel(f"GT\n{label}", fontsize=12)
    axarr[row_pred, 0].set_ylabel(f"Pred\n{label}", fontsize=12)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Adjust colorbar vertical placement per row group
    bar_height = 0.18
    bar_top = 0.94 - 0.24 * (row_pred // 2)
    cbar_ax = f.add_axes([0.88, bar_top - bar_height, 0.015, bar_height])
    f.colorbar(sm, cax=cbar_ax)


def parse_args() -> ArgsNamespace:
    parser = ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True, help="Name of the current run.")
    parser.add_argument("--dataset", type=str, default='shearflow', help="Name of dataset.")
    parser.add_argument("--random-seed", type=int, default=1543, help="Random seed.")
    parser.add_argument("--probpath_option", type=str, default='ours', help="Options for choosing probability path and vector field.")
    parser.add_argument("--train_option", type=str, default='separate', help="Options for choosing training scheme.")
    parser.add_argument("--ae_option", type=str, default='ae', help="Options for choosing autoencoders.")
    parser.add_argument("--solver", type=str, default='euler', help="Options for choosing sampler scheme.")
    parser.add_argument("--sigma", type=float, default=0.01, help="Sigma for our method.")
    parser.add_argument("--sigma_min", type=float, default=0.001, help="Sigma_min for our method.")
    parser.add_argument("--sigma_sam", type=float, default=0.0, help="Sigma_sam for our method.")
    parser.add_argument("--snapshots-per-sample", type=int, default=25, help="Number of snapshots per sample.")
    parser.add_argument("--condition-snapshots", type=int, default=5, help="Number of snapshots per sample.")
    parser.add_argument("--snapshots-to-generate", type=int, default=20, help="Number of snapshots per sample.") 

    # paths 
    parser.add_argument("--path_to_checkpoints", type=str, default='/home/ldr934/minFlowMatching/checkpoints/', help="Path to the checkpoints of pre-trained ae.")
    parser.add_argument("--path_to_results", type=str, default='/home/ldr934/minFlowMatching/results', help="Path to save the results.")

    # optimizer parameters
    parser.add_argument("--learning-rate", type=float, default=0.00005, help="Learning rate.") 
    parser.add_argument("--weight-decay", type=float, default=0, help="Weight decay.")
    
    # training parameters
    parser.add_argument("--epochs", type=int, default=10001, help="Number of epochs.")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Train batch size.")
    parser.add_argument("--test_batch_size", type=int, default=16, help="Test batch size.")

    # evaluation options
    parser.add_argument("--sampling_steps", type=int, default=5, help="Number of integration steps.")    
    parser.add_argument("--val-freq", type=int, default=1, help="Number of snapshots per sample.")    
    parser.add_argument("--plot-freq", type=int, default=5, help="Number of snapshots per sample.")
    parser.add_argument("--early_stopping_patience", type=int, default=6, help="Patience for early stopping.")

    parser.add_argument("--scale", type=int, default=16, help="Scale for the encoder and decoder.")
    
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
    train_dataset = ShearFlowDataset(data_dir=data_dir, split="train", snapshots_per_sample=args.snapshots_per_sample, shuffle=True)
    val_dataset = ShearFlowDataset(data_dir=data_dir, split="valid", snapshots_per_sample=args.snapshots_per_sample)
    test_dataset = ShearFlowDataset(data_dir=data_dir, split="test", snapshots_per_sample=args.snapshots_per_sample)

    # Debugging
    print(f"Length of train_dataset: {len(train_dataset)}")
    print(f"Length of val_dataset: {len(val_dataset)}")
    print(f"Length of test_dataset: {len(test_dataset)}")
    print(f"Sample shape: {train_dataset[0].shape}")

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True) #, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)

elif args.dataset == 'simpleflow':
    in_channels = 1 
    out_channels = 1
    state_size = 4
    enc_mid_channels = 64
    dec_mid_channels = 128
    state_res = [8,8]
    datasets = {}
    for key in ["train", "val"]:
        datasets[key] = SimpleFlowDataset(snapshots_per_sample=args.snapshots_per_sample)
    datasets["test"] = EvalSimpleFlowDataset(snapshots_per_sample=args.snapshots_per_sample)

    train_loader = DataLoader(dataset=datasets['train'], batch_size=args.train_batch_size, 
                            shuffle=True, num_workers=4)

    val_loader = DataLoader(dataset=datasets['val'], batch_size=args.train_batch_size,
            shuffle=False, num_workers=4)

    test_loader = DataLoader(dataset=datasets['test'], batch_size=args.test_batch_size,
            shuffle=False, num_workers=4)
else:
    raise ValueError('Invalid dataset option!')

# Setup losses
flow_matching_mse_loss_fun = nn.MSELoss()
ae_mse_loss_fun = nn.MSELoss()  
val_mse_loss_fun = nn.MSELoss()  
test_mse_loss_fun = nn.MSELoss()


# Setup model and distribute across gpus
if args.train_option == "end-to-end":
    model = Model_EndtoEnd(state_size=state_size, in_channels=in_channels, out_channels=out_channels, enc_mid_channels=enc_mid_channels, dec_mid_channels=dec_mid_channels, state_res=state_res, ours_sigma=args.sigma, sigma_min=args.sigma_min, sigma_sam=args.sigma_sam)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

elif args.train_option == "separate":
    print("Loading AE model...")
    AE_PATH = args.path_to_checkpoints + "ae_" + args.dataset + "_best_data_subset.pt"
    ae_model = Model_AE(state_size=state_size, in_channels=in_channels, out_channels=out_channels, enc_mid_channels=enc_mid_channels, dec_mid_channels=dec_mid_channels)
    ae_model.load_state_dict(torch.load(AE_PATH, map_location=device))
    ae_model.to(device)
    #AE_PATH = args.path_to_checkpoints + args.dataset + ".pt"
    #ae_model = torch.load(AE_PATH)
    ae_model.eval()
    if args.ae_option == "ae":
        model = Model(ae_model.encoder, ae_model.decoder, state_size=state_size, state_res=state_res, ours_sigma=args.sigma, sigma_min=args.sigma_min, sigma_sam=args.sigma_sam)
    else:
        raise ValueError('Invalid AE option!')

    model.to(device) 

    total_params_ae = sum(p.numel() for p in ae_model.parameters() if p.requires_grad)
    print('# AE parameters: ', total_params_ae)

    total_params_flow = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('# Trainable parameters for flow matching model: ', total_params_flow)

elif args.train_option == "data-space":
    print("Loading AE model...")
    ae_model = Model_AE_Simple(state_size=state_size, in_channels=in_channels, out_channels=out_channels, scale=scale)
    model = Model(ae_model.encoder, ae_model.decoder, state_size=state_size, state_res=state_res, ours_sigma=args.sigma, sigma_min=args.sigma_min, sigma_sam=args.sigma_sam)
    model.to(device) 

    total_params_ae = sum(p.numel() for p in ae_model.parameters() if p.requires_grad)
    print('# AE parameters: ', total_params_ae)
    total_params_flow = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('# Trainable parameters for flow matching model: ', total_params_flow)

else:
    raise ValueError('Invalid training option!')


# Setup optimizer and scheduler if not yet
optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999))

lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
        num_warmup_steps=len(train_loader) * 5, # we need only a very shot warmup phase for our data
        num_training_steps=(len(train_loader) * args.epochs))



# Create directory to save model checkpoints
BEST_FM_PATH = os.path.join(args.path_to_checkpoints)
os.makedirs(BEST_FM_PATH, exist_ok=True)
best_fm_state_dict = None
best_val_loss = float('inf')
patience_counter = 0




# Start training
print("Starting training...")
for epoch in range(args.epochs):

    # ---------------- TRAINING ----------------
    train_start = time.perf_counter()

    model.train()
    train_gen = tqdm(train_loader, desc="Training")

    total_loss = 0.0
    total_flow_loss = 0.0
    total_ae_loss = 0.0
    num_batches = 0

    N = 0
    for batch in train_gen:
        if N >= 1:  
            break
        N += 1
        # Fetch data
        observations = batch.cuda()
        batch_size = observations.size(0)
     
        if args.train_option == "end-to-end":
            input_snapshots, reconstructed_snapshots, target_vectors, reconstructed_vectors = model(observations, option=args.probpath_option)
            flow_matching_loss = flow_matching_mse_loss_fun(target_vectors, reconstructed_vectors)
            ae_loss = ae_mse_loss_fun(input_snapshots, reconstructed_snapshots)
            loss = flow_matching_loss + ae_loss

        elif args.train_option == "separate":
            target_vectors, reconstructed_vectors = model(observations, option=args.probpath_option)
            flow_matching_loss = flow_matching_mse_loss_fun(target_vectors, reconstructed_vectors)
            ae_loss = 0.0
            loss = flow_matching_loss 

        elif args.train_option == "data-space":
            input_snapshots = observations[:, :args.condition_snapshots]
            target_vectors, reconstructed_vectors = model(input_snapshots, option=args.probpath_option)
            flow_matching_loss = flow_matching_mse_loss_fun(target_vectors, reconstructed_vectors)
            ae_loss = 0.0
            loss = flow_matching_loss

        
     
        # Backward pass
        model.zero_grad()
        loss.backward()
    
        # Optimizer step
        optimizer.step()

        # Update loss
        total_loss += loss.item()
        total_flow_loss += flow_matching_loss.item()
        total_ae_loss += ae_loss if isinstance(ae_loss, float) else ae_loss.item()
        num_batches += 1
    
    # update learning rate
    lr_scheduler.step()
    
    # Close loading bar
    #train_gen.close()

    # Compute average loss
    avg_loss = total_loss / num_batches
    avg_flow_loss = total_flow_loss / num_batches
    avg_ae_loss = total_ae_loss / num_batches

    print(f"Epoch={epoch}: Avg. test loss (MSE) = {avg_loss}, avg. flow loss = {avg_flow_loss}, avg. ae loss = {avg_ae_loss}")

    train_end = time.perf_counter()
    print(f"Epoch {epoch} training time: {train_end - train_start:.2f} seconds")



    # ---------------- VALIDATION ----------------
    if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
        val_start = time.perf_counter()

        model.eval()
        # Setup loading bar
        val_gen = tqdm(val_loader, desc="Validation")
        val_losses = []
        
        N = 0
        for batch in val_gen:
            if N >= 2:  
                break
            N += 1
            # Fetch data
            observations = batch.cuda()
            condition_snapshots = observations[:, :args.condition_snapshots]
            targets = observations[:,args.condition_snapshots:]
            # Log media
            generated_observations, all_latent_trajectories = model.plot_latens(
                observations=condition_snapshots,
                num_condition_snapshots=args.condition_snapshots, 
                num_snapshots=args.snapshots_to_generate, 
                steps=args.sampling_steps, 
                option=args.probpath_option)
            


    sample_idx = 0
    n_channels = state_size
    n_plot_steps = 5

    # Select trajectory: [steps, c, h, w]
    latent_trajectory = all_latent_trajectories[:, 0, sample_idx]  # shape: [steps, c, h, w]
   

    # Evenly select time steps
    step_indices = torch.linspace(0, latent_trajectory.size(0) - 1, n_plot_steps).long()

    # Reserve extra column for colorbars
    fig, axes = plt.subplots(
        n_channels, n_plot_steps,
        figsize=(2 * n_plot_steps, n_channels),
        gridspec_kw={'width_ratios': [1] * n_plot_steps}  # Equal widths
    )

    if n_channels == 1:
        axes = axes[None, :]  # Ensure 2D

    if n_channels == 4:
        plot_order = [1, 0, 2, 3] 
    elif n_channels == 8:
        plot_order = [0, 1, 2, 3, 4, 5, 6, 7]

    for row, i in enumerate(plot_order):
        images = [latent_trajectory[step.item(), i].cpu().numpy() for step in step_indices]
        absmax = max(abs(img).max() for img in images)
        vmin, vmax = -absmax, absmax

        for col, step in enumerate(step_indices):
            img = images[col]
            ax = axes[row, col]
            im = ax.imshow(img, cmap="RdBu", vmin=vmin, vmax=vmax)

            if row == 0:
                ax.set_title(f"Step {step.item():.0f}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"Ch{i}", rotation=0, labelpad=30, fontsize=10, va='center')

            ax.set_xticks([])
            ax.set_yticks([])

        # Add colorbar in a fixed location outside plot grid (no size distortion)
        divider = make_axes_locatable(axes[row, -1])
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im, cax=cax)

    plt.suptitle(f"Latent evolution: {n_channels} channels × 5 integration steps", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.95, 0.97])  # Leave space on the right
    plt.subplots_adjust(hspace=0.2, wspace=0.1)

    if args.train_option == "data-space":
        plt.savefig(f"plots_ae_latents/Latent_evolution_data_space_scale_{args.scale}.pdf")
    else:
        plt.savefig("plots_ae_latents/Latent_evolution.pdf")
    plt.show()

    last_condition = condition_snapshots[sample_idx, -1]  # shape: [C, H, W]

    n_display_channels = 4  # number of channels to display
    last_condition = condition_snapshots[sample_idx, -1]  # shape: [C, H, W]

    fig, axes = plt.subplots(1, n_display_channels, figsize=(10, 2))
    axes[0], axes[1] = axes[1], axes[0]

    titles = ["p", "s", "$v_x$", "$v_y$"]

    for i in range(n_display_channels):
        img = last_condition[i].cpu().numpy()
        absmax = abs(img).max()
        ax = axes[i]
        im = ax.imshow(img, cmap="RdBu", vmin=-absmax, vmax=absmax)
        ax.set_title(f"{titles[i]}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    plt.suptitle("Last conditioning frames", fontsize=14)
    plt.tight_layout()
    if args.train_option == "data-space":
        plt.savefig(f"plots_ae_latents/Last_condition_frames_data_space_scale_{args.scale}.pdf")
    else:
        plt.savefig("plots_ae_latents/Last_condition_frames.pdf")
    plt.show()

    



    fig, axes = plt.subplots(n_display_channels, 1, figsize=(2.5, 5))
    axes[0], axes[1] = axes[1], axes[0]

    for i in range(n_display_channels):
        img = last_condition[i].cpu().numpy()
        absmax = abs(img).max()
        ax = axes[i]
        im = ax.imshow(img, cmap="RdBu", vmin=-absmax, vmax=absmax)
        #ax.set_title(f"{titles[i]}", fontsize=10)
        ax.set_ylabel(f"{titles[i]}", rotation=0, labelpad=30, fontsize=10, va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    plt.tight_layout()
    plt.savefig("plots_ae_latents/Last_condition_frames_vertical.pdf")
    plt.show()
