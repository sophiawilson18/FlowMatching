import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T
import os
import yaml
import random



import torch
from torch.utils.data import Dataset
import h5py
import os
import numpy as np

class ShearFlowDataset(Dataset):
    def __init__(self, data_dir, split="train", snapshots_per_sample=4, mode="one-step"):
        """
        Custom dataset to load HDF5 files for shear flow dataset.
        
        Args:
        - data_dir (str): Path to dataset directory containing 'train', 'valid', 'test' folders.
        - split (str): One of ['train', 'valid', 'test'].
        - snapshots_per_sample (int): Number of time steps per sample.
        """

        self.snapshots_per_sample = snapshots_per_sample
        self.split = split
        self.data_files = sorted([os.path.join(data_dir, split, f) for f in os.listdir(os.path.join(data_dir, split)) if f.endswith(".hdf5")])

        # Store metadata
        self.trajectories_per_file = 32 if split == "train" else 4
        self.time_steps = 200
        self.channels = 4
        self.resolution = (256, 512)
        self.mode = mode

        # Compute total number of samples
        self.total_trajectories = len(self.data_files) * self.trajectories_per_file
        self.total_samples = self.total_trajectories * (self.time_steps - snapshots_per_sample)

        # Load normalization stats
        self.stats_path = "/home/ldr934/TheWell/the_well-original/the_well/datasets/shear_flow/stats.yaml"
        self._load_normalization_stats()


    def _load_normalization_stats(self):
        with open(self.stats_path, "r") as f:
            stats = yaml.safe_load(f)

        # Convert means to tensors
        self.means = {
            "pressure": torch.tensor(stats["mean"]["pressure"], dtype=torch.float32),
            "tracer": torch.tensor(stats["mean"]["tracer"], dtype=torch.float32),
            "velocity_x": torch.tensor(stats["mean"]["velocity"][0], dtype=torch.float32),
            "velocity_y": torch.tensor(stats["mean"]["velocity"][1], dtype=torch.float32),
        }

        # Convert stds to tensors with clamping
        self.stds = {
            "pressure": torch.tensor(stats["std"]["pressure"], dtype=torch.float32),
            "tracer": torch.tensor(stats["std"]["tracer"], dtype=torch.float32),
            "velocity_x": torch.tensor(stats["std"]["velocity"][0], dtype=torch.float32),
            "velocity_y": torch.tensor(stats["std"]["velocity"][1], dtype=torch.float32),
        }

    def __len__(self):
        if self.mode == "rollout":
            return self.total_trajectories
        else:
            return self.total_samples  # rolling window

    def __getitem__(self, index):
        if self.mode == "rollout":
            file_idx = index // self.trajectories_per_file
            traj_idx = index % self.trajectories_per_file
            time_idx = 0
            length = self.snapshots_per_sample
        else:
            file_idx = index // (self.trajectories_per_file * (self.time_steps - self.snapshots_per_sample))
            traj_idx = (index // (self.time_steps - self.snapshots_per_sample)) % self.trajectories_per_file
            if self.split == "train":
                time_idx = random.randint(0, self.time_steps - self.snapshots_per_sample - 1)
            else:
                time_idx = index % (self.time_steps - self.snapshots_per_sample)
            length = self.snapshots_per_sample

        with h5py.File(self.data_files[file_idx], "r", swmr=True) as f:
            pressure = f["t0_fields"]["pressure"][traj_idx, time_idx:time_idx + length]
            tracer = f["t0_fields"]["tracer"][traj_idx, time_idx:time_idx + length]
            velocity = f["t1_fields"]["velocity"][traj_idx, time_idx:time_idx + length]

            # Split velocity into separate channels
            velocity_x = velocity[..., 0]  # (T, 256, 512)
            velocity_y = velocity[..., 1]  # (T, 256, 512)

            # Convert to Torch tensors before normalization
            pressure = torch.as_tensor(pressure, dtype=torch.float32)
            tracer = torch.as_tensor(tracer, dtype=torch.float32)
            velocity_x = torch.as_tensor(velocity_x, dtype=torch.float32)
            velocity_y = torch.as_tensor(velocity_y, dtype=torch.float32)

            # Normalize fields
            pressure = (pressure - self.means["pressure"]) / self.stds["pressure"]
            tracer = (tracer - self.means["tracer"]) / self.stds["tracer"]
            velocity_x = (velocity_x - self.means["velocity_x"]) / self.stds["velocity_x"]
            velocity_y = (velocity_y - self.means["velocity_y"]) / self.stds["velocity_y"]

            # Stack all channels
            sample = torch.empty((length, self.channels, 256, 512), dtype=torch.float32)
            sample[:, 0] = pressure
            sample[:, 1] = tracer
            sample[:, 2] = velocity_x
            sample[:, 3] = velocity_y

        return sample
    
    def denormalize(self, sample):
        """
        Denormalize sample with shape [..., 4, H, W]
        """
        if sample.dim() < 3 or sample.size(-3) != 4:
            raise ValueError(f"Expected input with 4 channels, got shape {sample.shape}")

        denorm = sample.clone()
        denorm[..., 0, :, :] = denorm[..., 0, :, :] * self.stds["pressure"] + self.means["pressure"]
        denorm[..., 1, :, :] = denorm[..., 1, :, :] * self.stds["tracer"] + self.means["tracer"]
        denorm[..., 2, :, :] = denorm[..., 2, :, :] * self.stds["velocity_x"] + self.means["velocity_x"]
        denorm[..., 3, :, :] = denorm[..., 3, :, :] * self.stds["velocity_y"] + self.means["velocity_y"]

        return denorm
    






class NavierStokesDataset(Dataset):

    def __init__(self, snapshots_per_sample=5):

        self.snapshots_per_sample = snapshots_per_sample
        path = "/data/" #CHANGE THIS TO YOUR PATH
        h5_file = h5py.File(os.path.join(path, "ns_incom_inhom_2d_512-0.h5"), "r")
        #print(list(h5_file.keys())) #['force', 'particles', 't', 'velocity']
        data = np.array(h5_file['velocity'])  # (4, 1000, 512, 512, 2)
        our_data = data[0]  

        print('our data shape', our_data.shape)
       
        h5_file.close()
        
        self.flow = our_data[:800,:,:,:]

    def __len__(self):
        return 600

    def max_index(self):
        return 600

    def __getitem__(self, index, time_idx=0):
        
        prefinals = []
        for i in range(index, index + self.snapshots_per_sample):
                prefinals.append(torch.Tensor(self.flow[i]).float())

        data = torch.stack(prefinals)

        data = data.permute(0, 3, 1, 2)

        return data


class EvalNavierStokesDataset(Dataset):

    def __init__(self, snapshots_per_sample=5):

        self.snapshots_per_sample = snapshots_per_sample
        path = "/data/" #CHANGE THIS TO YOUR PATH
        h5_file = h5py.File(os.path.join(path, "ns_incom_inhom_2d_512-0.h5"), "r")
        print(list(h5_file.keys())) #['force', 'particles', 't', 'velocity']
        data = np.array(h5_file['velocity'])  # (4, 1000, 512, 512, 2)
        our_data = data[0]  
        h5_file.close()
        
        self.flow = our_data[800:1000,:,:,:]

    def __len__(self):
        return 1 

    def max_index(self):
        return 1

    def __getitem__(self, index, time_idx=0):
        
        prefinals = []
        for i in range(index, index + self.snapshots_per_sample):
                prefinals.append(torch.Tensor(self.flow[i]).float())

        data = torch.stack(prefinals)

        data = data.permute(0, 3, 1, 2)

        return data


class ReacDiffDataset(Dataset):

    def __init__(self, snapshots_per_sample=5):

        self.snapshots_per_sample = snapshots_per_sample
        path = "/data/" #CHANGE THIS TO YOUR PATH
        h5_file = h5py.File(os.path.join(path, "2D_diff-react_NA_NA.h5"), "r")
        num_samples = len(h5_file.keys())
        seed = 0 #np.random.randint(0, num_samples) 
        seed = str(seed).zfill(4)
        data = np.array(h5_file[f"{seed}/data"], dtype="f")  # dim = [101, 128, 128, 2]
        print('original data shape', data.shape)
        h5_file.close()
        
        self.flow = data[:80,:,:,:]

    def __len__(self):
        return 60

    def max_index(self):
        return 60

    def __getitem__(self, index, time_idx=0):
        
        prefinals = []
        for i in range(index, index + self.snapshots_per_sample):
                prefinals.append(torch.Tensor(self.flow[i]).float())

        data = torch.stack(prefinals)

        data = data.permute(0, 3, 1, 2)

        return data


class EvalReacDiffDataset(Dataset):

    def __init__(self, snapshots_per_sample=5):

        self.snapshots_per_sample = snapshots_per_sample
        path = "/data/" #CHANGE THIS TO YOUR PATH
        h5_file = h5py.File(os.path.join(path, "2D_diff-react_NA_NA.h5"), "r")
        num_samples = len(h5_file.keys())
        seed = 0 #np.random.randint(0, num_samples) 
        seed = str(seed).zfill(4)
        data = np.array(h5_file[f"{seed}/data"], dtype="f")  # dim = [101, 128, 128, 2]
        h5_file.close()
        
        self.flow = data[80:100,:,:,:]

    def __len__(self):
        return 1 

    def max_index(self):
        return 1

    def __getitem__(self, index, time_idx=0):
        
        prefinals = []
        for i in range(index, index + self.snapshots_per_sample):
                prefinals.append(torch.Tensor(self.flow[i]).float())

        data = torch.stack(prefinals)

        data = data.permute(0, 3, 1, 2)
        print('original data shape', data.shape)

        return data


class ShalloWaterDataset(Dataset):

    def __init__(self, snapshots_per_sample=5):

        self.snapshots_per_sample = snapshots_per_sample
        path = "/data/2D/shallow-water" #CHANGE THIS TO YOUR PATH
        h5_file = h5py.File(os.path.join(path, "2D_rdb_NA_NA.h5"), "r")
        num_samples = len(h5_file.keys())
        seed = 0 #np.random.randint(0, num_samples) 
        seed = str(seed).zfill(4)
        data = np.array(h5_file[f"{seed}/data"], dtype="f")  # dim = [101, 128, 128, 1]

        mean = np.mean(data)
        std = np.std(data)
        standardized_data = (data - mean) / std

        # Min-max scaling to [-1, 1]
        min_val = np.min(standardized_data)
        max_val = np.max(standardized_data)

        # Normalize to [-1, 1]
        scaled_data = 2 * (standardized_data - min_val) / (max_val - min_val) - 1

        print('original data shape', data.shape)
        print('mean: ', mean)
        print('stdev: ', std)
        h5_file.close()
        
        self.flow = scaled_data[:80,:,:,:]

    def __len__(self):
        return 60

    def max_index(self):
        return 60

    def __getitem__(self, index, time_idx=0):
        
        prefinals = []
        for i in range(index, index + self.snapshots_per_sample):
                prefinals.append(torch.Tensor(self.flow[i]).float())

        data = torch.stack(prefinals)

        data = data.permute(0, 3, 1, 2)

        return data


class EvalShallowWaterDataset(Dataset):

    def __init__(self, snapshots_per_sample=5):

        self.snapshots_per_sample = snapshots_per_sample
        path = "/data/2D/shallow-water" #CHANGE THIS TO YOUR PATH
        h5_file = h5py.File(os.path.join(path, "2D_rdb_NA_NA.h5"), "r")
        num_samples = len(h5_file.keys())
        seed = 0 #np.random.randint(0, num_samples) 
        seed = str(seed).zfill(4)
        data = np.array(h5_file[f"{seed}/data"], dtype="f")  # dim = [101, 128, 128, 1]
       
        mean = np.mean(data)
        std = np.std(data)
        standardized_data = (data - mean) / std

        # Min-max scaling to [-1, 1]
        min_val = np.min(standardized_data)
        max_val = np.max(standardized_data)

        # Normalize to [-1, 1]
        scaled_data = 2 * (standardized_data - min_val) / (max_val - min_val) - 1

        h5_file.close()
        
        self.flow = scaled_data[80:100,:,:,:]

    def __len__(self):
        return 1 

    def max_index(self):
        return 1

    def __getitem__(self, index, time_idx=0):
        
        prefinals = []
        for i in range(index, index + self.snapshots_per_sample):
                prefinals.append(torch.Tensor(self.flow[i]).float())

        data = torch.stack(prefinals)

        data = data.permute(0, 3, 1, 2)

        return data


class SimpleFlowDataset(Dataset):

    def __init__(self, snapshots_per_sample=5):

        self.snapshots_per_sample = snapshots_per_sample

        # Read dataset
        data = np.load("/home/ldr934/minFlowMatching/data/small_flow.npy") 

        

        print('original data shape', data.shape)   # (151, 64, 64)
        data = data.reshape(data.shape[0],64,64,1) # (151, 64, 64, 1)
        self.flow = data[:125,:,:,:]               # (125, 64, 64, 1) for training 

    def __len__(self):
        return 100

    def max_index(self):
        return 100

    def __getitem__(self, index, time_idx=0):
        # Fetch the data
        
        prefinals = []
        for i in range(index, index + self.snapshots_per_sample):
                prefinals.append(torch.Tensor(self.flow[i]).float())

        data = torch.stack(prefinals)

        data = data.permute(0, 3, 1, 2) # (5, 1, 64, 64)

        return data


class EvalSimpleFlowDataset(Dataset):

    def __init__(self, snapshots_per_sample=5):

        self.snapshots_per_sample = snapshots_per_sample

        # Read dataset
        data = np.load('/home/ldr934/minFlowMatching/data/small_flow.npy') 

        data = data.reshape(data.shape[0],64,64,1)
        self.flow = data[125:150,:,:,:]

    def __len__(self):
        return 1 

    def max_index(self):
        return 1

    def __getitem__(self, index, time_idx=0):
        
        prefinals = []
        for i in range(index, index + self.snapshots_per_sample):
                prefinals.append(torch.Tensor(self.flow[i]).float())

        data = torch.stack(prefinals)

        data = data.permute(0, 3, 1, 2)

        return data
