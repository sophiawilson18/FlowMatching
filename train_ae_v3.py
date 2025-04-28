import os
#import wandb

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

from model.model_ae_v3 import AE_v3
from utils.get_data import SimpleFlowDataset, EvalSimpleFlowDataset, ShearFlowDataset

import torch.optim.lr_scheduler as lr_scheduler
from einops import rearrange
from transformers import get_cosine_schedule_with_warmup
from argparse import ArgumentParser, Namespace as ArgsNamespace
from copy import deepcopy
import csv

torch.cuda.empty_cache()

def print_num_params(model: nn.Module):
    print(f"{'Module':40} | {'# Params':>10}")
    print("-" * 55)
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num = param.numel()
            print(f"{name:40} | {num:10,}")
            total_params += num
    print("-" * 55)
    print(f"{'Total trainable parameters':40} | {total_params:10,}")


def parse_args() -> ArgsNamespace:
    parser = ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True, help="Name of the current run.")
    parser.add_argument("--dataset", type=str, default='shearflow', help="Name of dataset.")
    parser.add_argument("--random-seed", type=int, default=1543, help="Random seed.")
    parser.add_argument("--ae_option", type=str, default='ae_v3', help="Options for choosing autoencoders.")

    # optimizer parameters
    parser.add_argument("--ae_learning_rate", type=float, default=1e-4, help="Learning rate for AE.") 
    parser.add_argument("--ae_weight_decay", type=float, default=1e-4, help="Weight decay for AE.")
    parser.add_argument("--ae_lr_scheduler", type=str, default='cosine', help="Options for learning rate scheduler for AE.")
    parser.add_argument("--ae_loss", type=str, default='mse', help="Loss function for AE.")
    
    # training parameters
    parser.add_argument("--ae_epochs", type=int, default=10001, help="Number of epochs for training ae, if trained separately.")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Train batch size.")
    parser.add_argument("--test_batch_size", type=int, default=8, help="Test batch size.")

    parser.add_argument("--snapshots-per-sample", type=int, default=25, help="Number of snapshots per sample.")

    # SW: Add this line to support the checkpoint path argument
    parser.add_argument("--path_to_ae_checkpoints", type=str, default="./ae/checkpoints", help="Path to save AE model checkpoints.")
    parser.add_argument("--scale", type=int, default=16, help="Scale for the encoder and decoder.")


    return parser.parse_args()

args = parse_args()


# Launch processes.
print('Launching processes...')

#wandb.login(key="9b17d67ba9f4c654b23d497c8899c4cd1a7a65b4")
#
#run = wandb.init(
#    # Set the project where this run will be logged
#    project="minFlowMatching_ae_pretraining",
#    name=args.run_name,
#    # Track hyperparameters and run metadata
#    config={
#        "learning_rate": args.ae_learning_rate,
#        "epochs": args.ae_epochs,
#    },
#)

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
    enc_mid_channels = 96
    dec_mid_channels = 192
    state_res = [16,32] 
    state_size = 8

    data_dir = "/home/ldr934/TheWell/the_well-original/the_well/datasets/shear_flow/data"
    train_dataset = ShearFlowDataset(data_dir=data_dir, split="train", snapshots_per_sample=args.snapshots_per_sample)
    val_dataset = ShearFlowDataset(data_dir=data_dir, split="valid", snapshots_per_sample=args.snapshots_per_sample)
    test_dataset = ShearFlowDataset(data_dir=data_dir, split="test", snapshots_per_sample=args.snapshots_per_sample)

    # Debugging
    print(f"Length of train_dataset: {len(train_dataset)}")
    print(f"Length of val_dataset: {len(val_dataset)}")
    print(f"Length of test_dataset: {len(test_dataset)}")
    print(f"Sample shape: {train_dataset[0].shape}")

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
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
if args.ae_loss == "mse":
    ae_mse_loss_fun = nn.MSELoss()  


ae_epochs = args.ae_epochs 
if args.ae_option == "ae":
    ae_model = Model_AE(state_size=state_size, in_channels=in_channels, out_channels=out_channels, enc_mid_channels=enc_mid_channels, dec_mid_channels=dec_mid_channels)

elif args.ae_option == "ae_v3":
    ae_model = AE_v3(in_channels=in_channels, out_channels=out_channels, scale=args.scale)

else:
    raise ValueError('Invalid ae option!')

total_params_ae = sum(p.numel() for p in ae_model.parameters() if p.requires_grad)
print('# trainable parameters: ', total_params_ae)
#print_num_params(ae_model)

ae_model.to(device)
ae_optimizer = torch.optim.AdamW(
        ae_model.parameters(),
        lr=args.ae_learning_rate,
        weight_decay=args.ae_weight_decay,
        betas=(0.9, 0.999))

if args.ae_lr_scheduler == "exponential":
    ae_lr_scheduler = lr_scheduler.ExponentialLR(ae_optimizer, gamma=0.999)
elif args.ae_lr_scheduler == "cosine":
    ae_lr_scheduler = get_cosine_schedule_with_warmup(ae_optimizer, 
    num_warmup_steps=len(train_loader) * 5, # we need only a very shot warmup phase for our data
    num_training_steps=(len(train_loader) * ae_epochs))
    #ae_lr_scheduler = lr_scheduler.CosineAnnealingLR(ae_optimizer, T_max=10, eta_min=0)


# --- ADDED: Early stopping setup ---
BEST_AE_PATH = os.path.join(args.path_to_ae_checkpoints, f"AE_best.pt")
best_val_loss = float("inf")
best_ae_state_dict = None
patience = 6
patience_counter = 0
# -----------------------------------

train_losses = []
val_losses = []

for epoch in range(ae_epochs):
    ae_model.train()
    train_gen = tqdm(train_loader, desc="Training")
    
    total_loss = 0.0
    batch_count = 0

    for batch in train_gen:
        # Fetch data
        observations = batch.cuda()

        if args.ae_option == "ae":
            input_snapshots, reconstructed_snapshots = ae_model(observations)
            ae_loss = ae_mse_loss_fun(input_snapshots, reconstructed_snapshots)

        if args.ae_option == "ae_v3":
            # [B, T, C, H, W] → [B*T, C, H, W]
            observations = rearrange(observations, "b t c h w -> (b t) c h w")

            # AE forward pass
            input_snapshots, reconstructed_snapshots = ae_model(observations)

            # Compute loss
            ae_loss = ae_mse_loss_fun(reconstructed_snapshots, input_snapshots)
        else:
            raise ValueError('Invalid ae option!')
        
        total_loss += ae_loss.item()
        batch_count += 1

        # Backward pass
        ae_model.zero_grad()
        ae_loss.backward()
    
        # Optimizer step
        ae_optimizer.step()
    
    # update learning rate
    ae_lr_scheduler.step()

    # Optional: clean up GPU memory between epochs
    torch.cuda.empty_cache()

    # Log training loss
    epoch_loss = total_loss / batch_count
    train_losses.append(epoch_loss)

    print(f"Epoch = {epoch}, train loss = {epoch_loss:.6f}")
    #wandb.log({"Train loss": epoch_loss})

    # --- ADDED: Validation + early stopping ---
    ae_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_batch in val_loader:
            val_batch = val_batch.cuda()
            input_snapshots, reconstructed_snapshots = ae_model(val_batch)
            loss = ae_mse_loss_fun(input_snapshots, reconstructed_snapshots)
            val_loss += loss.item()

    val_loss = val_loss / max(1, len(val_loader))
    val_losses.append(val_loss)
    print(f"Epoch = {epoch}, val loss = {val_loss:.6f}")
    #wandb.log({"Val loss": val_loss})

    # save checkpoints every third epoch
    if epoch % 3 == 0:
        checkpoint_path = os.path.join(args.path_to_ae_checkpoints, f"AE_epoch{epoch}.pt")
        torch.save(ae_model.state_dict(), checkpoint_path)
        print(f"✅ Saved checkpoint at epoch {epoch}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_ae_state_dict = deepcopy(ae_model.state_dict())
        patience_counter = 0
        torch.save(best_ae_state_dict, BEST_AE_PATH)
        print(f"✅ Saved best model (val_loss={val_loss:.6f})")
    else:
        patience_counter += 1
        print(f"⏳ No improvement for {patience_counter}/{patience} epochs")

    if patience_counter >= patience:
        print("⛔ Early stopping triggered")
        break
    # ---


print('✅  Done with training AE...')

# save train loss and val loss
train_losses = np.array(train_losses)
val_losses = np.array(val_losses)

with open(f'ae/results/train_val_loss.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(train_losses, val_losses))
print('✅  Train and Val losses saved.')

epochs = list(range(len(train_losses)))
plt.figure()
plt.plot(epochs, train_losses, 'o-', label='Train Loss')
plt.plot(epochs, val_losses, 'o-', label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title(f'AE')
plt.grid(True)
plt.xlim(-1, len(train_losses))
plt.yscale('log')


# save plot
plt.savefig(f'ae/results/train_val_loss.png')
print('✅  Plot saved.')

# Save final model (optional)
FINAL_PATH = os.path.join(args.path_to_ae_checkpoints, f"AE_final.pt")
torch.save(ae_model.state_dict(), FINAL_PATH)
print('✅  Final AE model saved.')
