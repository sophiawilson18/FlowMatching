"""
Train a flow matching model with various autoencoder options on shearflow (or simpleflow) datasets.
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
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt

from model.fm import FM_model
from model.ae_fixed import AE_fixed
from model.ae_trainable import AE_trainable
from model.endtoend_ae_trainable import Model_EndtoEnd 
from utils.get_data import SimpleFlowDataset, EvalSimpleFlowDataset, ShearFlowDataset


# ---------------- Argument Parsing ----------------
def parse_args() -> ArgsNamespace:
    parser = ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True, help="Name of the current run.")
    parser.add_argument("--dataset", type=str, default='shearflow', help="Name of dataset.")
    parser.add_argument("--random-seed", type=int, default=41, help="Random seed.") 
    parser.add_argument("--probpath_option", type=str, default='ours', help="Options for choosing probability path and vector field.")
    parser.add_argument("--train_option", type=str, default='separate', help="Options for choosing training scheme.")
    parser.add_argument("--sigma", type=float, default=0.01, help="Sigma for our method.")
    parser.add_argument("--sigma_min", type=float, default=0.001, help="Sigma_min for our method.")
    parser.add_argument("--sigma_sam", type=float, default=0.0, help="Sigma_sam for our method.")
    parser.add_argument("--snapshots-per-sample", type=int, default=25, help="Number of snapshots per sample in the dataset.")
    parser.add_argument("--condition-snapshots", type=int, default=5, help="Number of condition snapshots for the model.")
    parser.add_argument("--snapshots-to-generate", type=int, default=20, help="Number of snapshots to generate for each sample during evaluation.")

    # paths 
    parser.add_argument("--path_to_fm_checkpoints", type=str, default='/home/ldr934/FlowMatching/checkpoints/', help="Path to save FM model checkpoints.")
    parser.add_argument("--path_to_ae_checkpoints", type=str, default="./ae/checkpoints", help="Path with saved AE model checkpoints.")
    parser.add_argument("--path_to_results", type=str, default='/home/ldr934/FlowMatching/results', help="Path to save the results.")
    
    # optimizer parameters
    parser.add_argument("--learning-rate", type=float, default=0.00005, help="Learning rate.") 
    parser.add_argument("--weight-decay", type=float, default=0, help="Weight decay.")
    
    # training parameters
    parser.add_argument("--epochs", type=int, default=10001, help="Number of epochs.")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Train batch size.")
    parser.add_argument("--test_batch_size", type=int, default=16, help="Test batch size.")

    # evaluation options
    parser.add_argument("--val-freq", type=int, default=1, help="Validation frequency.")
    parser.add_argument("--early_stopping_patience", type=int, default=6, help="Patience for early stopping.")

    # scale for the fixed autoencoder
    parser.add_argument("--scale", type=int, default=16, help="Scale for the encoder and decoder.")
    
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
# Shear Flow Dataset (training in latent and data space)
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

# Simple Flow Dataset (training in latent space)
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


# ---------------- Losses Setup ----------------
flow_matching_mse_loss_fun = nn.MSELoss()
ae_mse_loss_fun = nn.MSELoss()  
val_mse_loss_fun = nn.MSELoss()  


# ---------------- Model Setup ----------------
if args.train_option == "end-to-end": # end-to-end training with trainable autoencoder
    model = Model_EndtoEnd(state_size=state_size, in_channels=in_channels, out_channels=out_channels, enc_mid_channels=enc_mid_channels, dec_mid_channels=dec_mid_channels, state_res=state_res, ours_sigma=args.sigma, sigma_min=args.sigma_min, sigma_sam=args.sigma_sam)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

elif args.train_option == "separate": # separate training with trainable autoencoder
    print("Loading AE model...")
    AE_PATH = args.path_to_ae_checkpoints + "ae_" + args.dataset + ".pt"
    ae_model = AE_trainable(state_size=state_size, in_channels=in_channels, out_channels=out_channels, enc_mid_channels=enc_mid_channels, dec_mid_channels=dec_mid_channels)
    ae_model.load_state_dict(torch.load(AE_PATH, map_location=device))
    ae_model.to(device)
    ae_model.eval()
    model = FM_model(ae_model.encoder, ae_model.decoder, state_size=state_size, state_res=state_res, ours_sigma=args.sigma, sigma_min=args.sigma_min, sigma_sam=args.sigma_sam)
    model.to(device) 

    total_params_ae = sum(p.numel() for p in ae_model.parameters() if p.requires_grad)
    print('# AE parameters: ', total_params_ae)
    total_params_flow = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('# Trainable parameters for flow matching model: ', total_params_flow)

elif args.train_option == "data-space": # training in data space with fixed autoencoder
    print("Loading AE model...")
    ae_model = AE_fixed(state_size=state_size, in_channels=in_channels, out_channels=out_channels, scale=scale)
    model = FM_model(ae_model.encoder, ae_model.decoder, state_size=state_size, state_res=state_res, ours_sigma=args.sigma, sigma_min=args.sigma_min, sigma_sam=args.sigma_sam)
    model.to(device) 

    total_params_ae = sum(p.numel() for p in ae_model.parameters() if p.requires_grad)
    print('# AE parameters: ', total_params_ae)
    total_params_flow = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('# Trainable parameters for flow matching model: ', total_params_flow)

else:
    raise ValueError('Invalid training option!')


# ---------------- Optimizer and Scheduler Setup ----------------
optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999))

lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
        num_warmup_steps=len(train_loader) * 3, # we need only a very shot warmup phase for our data
        num_training_steps=(len(train_loader) * args.epochs))



# ---------------- Checkpoints and Early Stopping Setup ----------------
BEST_FM_PATH = os.path.join(args.path_to_fm_checkpoints)
os.makedirs(BEST_FM_PATH, exist_ok=True)
best_fm_state_dict = None
best_val_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []
epochs_val = [] 

# ---------------- Training Loop ----------------
print("Starting training...")
for epoch in range(args.epochs):
    model.train()
    train_gen = tqdm(train_loader, desc="Training")
    total_loss = 0.0

    for batch in train_gen:
        observations = batch.to(device)
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
    print(f"Epoch={epoch}: Avg. train loss (MSE) = {avg_loss}") 

    # Save training loss
    train_losses.append(avg_loss)

    train_gen.close()



    # ---------------- Validation loop ----------------
    if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
        model.eval()
        val_gen = tqdm(val_loader, desc="Validation")
        val_loss = 0.0
        
        for batch in val_gen:

            if args.train_option == "end-to-end":
                observations = batch.to(device)
                input_snapshots, reconstructed_snapshots, target_vectors, reconstructed_vectors = model(observations, option=args.probpath_option)
                val_flow_matching_loss = flow_matching_mse_loss_fun(target_vectors, reconstructed_vectors)
                val_ae_loss = ae_mse_loss_fun(input_snapshots, reconstructed_snapshots)
                val_mse = val_flow_matching_loss + val_ae_loss

            elif args.train_option == "separate":
                observations = batch.to(device)
                target_vectors, reconstructed_vectors = model(observations, option=args.probpath_option)
                val_mse = flow_matching_mse_loss_fun(target_vectors, reconstructed_vectors)

            elif args.train_option == "data-space":
                observations = batch.to(device)
                input_snapshots = observations[:, :args.condition_snapshots]
                target_vectors, reconstructed_vectors = model(input_snapshots, option=args.probpath_option)
                val_mse = val_mse_loss_fun(target_vectors, reconstructed_vectors)


            val_loss += val_mse.item()

        val_mse_mean = val_loss / len(val_loader)
        val_gen.close()
        print(f"Epoch={epoch}: Avg. validation (MSE) = {val_mse_mean}")

        # Save validation loss
        val_losses.append(val_mse_mean)
        epochs_val.append(epoch)

        # Early stopping
        if val_mse_mean < best_val_loss:
            best_val_loss = val_mse_mean
            patience_counter = 0
            best_fm_state_dict = deepcopy(model.state_dict())
            torch.save(best_fm_state_dict, os.path.join(BEST_FM_PATH, args.run_name + args.dataset + '.pt'))
        else:
            patience_counter += 1
            print(f"{epoch}: Validation loss did not improve from {best_val_loss}. Patience: {patience_counter}")
            if patience_counter >= args.early_stopping_patience:
                print("Early stopping activated.")
                break


# ---------------- Save Losses and Plot ----------------
train_losses = np.array(train_losses)
val_losses = np.array(val_losses)

directory = os.path.join(args.path_to_results, args.run_name)
os.makedirs(directory, exist_ok=True)

with open(os.path.join(directory, 'losses.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(zip(train_losses, val_losses))
print('✅  Train and Val losses saved.')

epochs_train = list(range(len(train_losses)))
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