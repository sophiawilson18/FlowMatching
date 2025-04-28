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


def add_field_plot(obs, gen, row_gt, row_pred, channel, label, args, cmap='RdBu'): # (12, 5, 4, 256, 512)
    global f, axarr, ncol  # ensure these are accessible

    # Compute symmetric color range
    data_gt = obs[0, :, channel]
    data_pred = gen[0, :, channel]
    vmax = np.max(np.abs(data_gt))
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
    parser.add_argument("--random-seed", type=int, default=41, help="Random seed.") # 1543
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
    parser.add_argument("--sampling_steps", type=int, default=10, help="Number of integration steps.")    
    parser.add_argument("--val-freq", type=int, default=1, help="Validation frequency.")
    parser.add_argument("--plot-freq", type=int, default=1, help="Plotting frequency.")
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
    train_dataset = ShearFlowDataset(data_dir=data_dir, split="train", snapshots_per_sample=args.snapshots_per_sample)
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

#print("Computing per-field mean and std of training data...")
#channel_sum = 0.0
#channel_squared_sum = 0.0
#total_pixels = 0
#
#for batch in train_loader:
#    # Assume batch has shape [B, C, H, W]
#    x = batch.to(torch.float32)  # [B, C, H, W]
#    print(f"Batch shape: {x.shape}")
#    B, T, C, H, W = x.shape
#    x = x.to(torch.float32).view(-1, C)
#    total_pixels += x.shape[0]
#    channel_sum += x.sum(dim=0)
#    channel_squared_sum += (x ** 2).sum(dim=0)
#
#mean = channel_sum / total_pixels
#std = (channel_squared_sum / total_pixels - mean ** 2).sqrt()
#print(f"Per-field mean: {mean}")
#print(f"Per-field std: {std}")

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
    model = Model(ae_model.encoder, ae_model.decoder, state_size=state_size, state_res=state_res, ours_sigma=args.sigma, sigma_min=args.sigma_min, sigma_sam=args.sigma_sam)
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
        num_warmup_steps=len(train_loader) * 3, # we need only a very shot warmup phase for our data
        num_training_steps=(len(train_loader) * args.epochs))



# Create directory to save model checkpoints
BEST_FM_PATH = os.path.join(args.path_to_checkpoints)
os.makedirs(BEST_FM_PATH, exist_ok=True)
best_fm_state_dict = None
best_val_loss = float('inf')
patience_counter = 0

train_losses = []
val_losses = []


# Start training
print("Starting training...")
for epoch in range(args.epochs):

    # ---------------- TRAINING ----------------
    #train_start = time.perf_counter()

    model.train()
    train_gen = tqdm(train_loader, desc="Training")

    total_loss = 0.0

    #N = 0
    for batch in train_gen:
        #if N >= 1:  
        #    break
        #N += 1


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
            loss = flow_matching_loss 

        elif args.train_option == "data-space":
            input_snapshots = observations[:, :args.condition_snapshots]
            target_vectors, reconstructed_vectors = model(input_snapshots, option=args.probpath_option)
            loss = flow_matching_mse_loss_fun(target_vectors, reconstructed_vectors)
     
        # Backward pass
        model.zero_grad()
        loss.backward()
    
        # Optimizer step
        optimizer.step()

        # Update loss
        total_loss += loss.item()

    
    # update learning rate
    lr_scheduler.step()

    # Compute average loss
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch={epoch}: Avg. train loss (MSE) = {avg_loss}") #, avg. flow loss = {avg_flow_loss}, avg. ae loss = {avg_ae_loss}")

    # Save training loss
    train_losses.append(avg_loss)

    #train_end = time.perf_counter()
    #print(f"Epoch {epoch} training time: {train_end - train_start:.2f} seconds")
    train_gen.close()



    # ---------------- VALIDATION ----------------
    if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
        #val_start = time.perf_counter()

        model.eval()
        # Setup loading bar
        val_gen = tqdm(val_loader, desc="Validation")
        val_loss = 0
        
        #N = 0
        for batch in val_gen:
            #if N >= 1:  
            #    break
            #N += 1

            observations = batch.cuda()
            input_snapshots = observations[:, :args.condition_snapshots]
            target_vectors, reconstructed_vectors = model(input_snapshots, option=args.probpath_option)
            val_mse = val_mse_loss_fun(target_vectors, reconstructed_vectors)
            val_loss += val_mse.item()

        val_mse_mean = val_loss / len(val_loader)
        val_gen.close()
        print(f"Epoch={epoch}: Avg. validation (MSE) = {val_mse_mean}")

        # Save validation loss
        val_losses.append(val_mse_mean)

        # Early stopping
        if val_mse_mean < best_val_loss:
            best_val_loss = val_mse_mean
            patience_counter = 0
            best_fm_state_dict = deepcopy(model.state_dict())
            torch.save(best_fm_state_dict, BEST_FM_PATH+args.run_name+'_FM_best.pt') 
        else:
            patience_counter += 1
            print(f"{epoch}: Validation loss did not improve from {best_val_loss}. Patience: {patience_counter}")
            if patience_counter >= args.early_stopping_patience:
                print("Early stopping activated.")
                break


        #.perf_counter()
        #print(f"Epoch {epoch} validation time: {val_end - val_start:.2f} seconds")




    #if epoch % args.plot_freq == 0 or epoch == args.epochs - 1:
    #    plot_start = time.perf_counter()
#
    #    for i, batch in enumerate(val_loader):
    #        if i >= 1:
    #            break
    #        # Fetch data
    #        observations = batch.cuda()
    #        condition_snapshots = observations[:, :args.condition_snapshots]
    #        targets = observations[:,args.condition_snapshots:]
    #        # Log media
    #        generated_observations = model.generate_snapshots(
    #            observations=condition_snapshots,
    #            num_condition_snapshots=args.condition_snapshots, 
    #            num_snapshots=args.snapshots_to_generate, 
    #            steps=args.sampling_steps, 
    #            option=args.probpath_option)
#
    #    # Plotting
    #    obs = test_dataset.denormalize(observations[0]).unsqueeze(0).cpu().numpy()
    #    gen = test_dataset.denormalize(generated_observations[0]).unsqueeze(0).cpu().numpy()
#
    #    obs = obs[0, -1, :, :, :]  # Select the first sample
    #    gen = gen[0, -1, :, :, :]
#
    #    # Create a figure with 2 rows and 4 columns
    #    f, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
#
    #    titles = ["s", "p", "$v_x$", "$v_y$"]
    #    plot_order = [1, 0, 2, 3]
#
    #    for col_idx, i in enumerate(plot_order):
    #        # Plot observation (top row)
    #        img_obs = obs[i]
    #        absmax = abs(img_obs).max()
    #        norm = mcolors.TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)
    #        im_obs = axes[0, col_idx].imshow(img_obs, cmap='RdBu', norm=norm)
    #        axes[0, col_idx].set_title(f"{titles[i]} (obs)", fontsize=12)
    #        axes[0, col_idx].axis("off")
#
    #        # Plot generation (bottom row)
    #        img_gen = gen[i]
    #        im_gen = axes[1, col_idx].imshow(img_gen, cmap='RdBu', norm=norm)
    #        axes[1, col_idx].set_title(f"{titles[i]} (gen)", fontsize=12)
    #        axes[1, col_idx].axis("off")
#
    #        # Add a horizontal colorbar underneath each column
    #        cbar = f.colorbar(im_gen, ax=[axes[0, col_idx], axes[1, col_idx]], orientation='horizontal', fraction=0.06, pad=0.1)
    #        cbar.ax.tick_params(labelsize=8)
#
    #    # Title and saving
    #    f.suptitle(f"Epoch {epoch}, Scale {args.scale}", fontsize=16)
    #    directory = os.path.join(args.path_to_results, args.run_name)
    #    os.makedirs(directory, exist_ok=True)
    #    savename = f'/evaluation_epoch={epoch}_scale={args.scale}.pdf'
    #    plt.savefig(directory + savename)
    #    plt.close('all')

# save train loss and val loss
train_losses = np.array(train_losses)
val_losses = np.array(val_losses)

directory = os.path.join(args.path_to_results, args.run_name)

if not os.path.exists(directory):
    os.makedirs(directory)

with open(os.path.join(directory, 'losses.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(zip(train_losses, val_losses))
print('✅  Train and Val losses saved.')

epochs_train = list(range(len(train_losses)))
epochs_val = list(range(0, args.epochs, args.val_freq))
if args.epochs - 1 not in epochs_val:
    epochs_val.append(args.epochs - 1)  # In case last val is at final epoch

plt.figure()
plt.plot(epochs_train, train_losses, 'o-', label='Train Loss')
plt.plot(epochs_val, val_losses, 'o-', label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title(f'Losses, scale {args.scale}')
plt.grid(True)
plt.xlim(-1, args.epochs)
plt.yscale('log')
plt.savefig(os.path.join(directory, 'losses.png'))
plt.close('all')


