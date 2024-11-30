# main.py
# Description: This file contains the main training loop for the model.

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.fft import fft, ifft
from torch.amp import GradScaler, autocast

# Utility imports
import os
import gc
import gzip
import matplotlib.pyplot as plt
import datetime
from typing import *

# Local imports
import utils as tools

class CodeGen(nn.Module):
    def __init__(self, K, l):
        super(CodeGen, self).__init__()
        hidden_size = 2 * K * l
        # Number of hidden layers = 2
        self.fc1 = nn.Linear(K * l, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, K * l)

    def forward(self, x):
        # Hidden layer activation function not specified in the paper
        # Using ReLU as the activation function
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # Output layer activation function = tanh
        return torch.tanh(self.fc3(x))

@torch.no_grad()
def get_samples(model, K, l, N, sigma=0.1, device='cuda'):
    '''
    Generate a N samples of K codes of length l.
    
    Parameters:
    -----------
    model (torch.nn.Module): Code generator model.
    K (int): Number of codes per code family.
    l (int): Code length.
    N (int): Number of samples to generate.
    sigma (float): Standard deviation of the proposal distribution.
    device (str): Device to run the model on.

    Returns:
    --------
    torch.Tensor: N code families of K codes of length l.
    torch.Tensor: N times ∇_θ*log(p_θ(x^i)). [Nx(K*l)]
    '''
    with torch.no_grad():
        # Generate stochastic input noise
        x = torch.randn(N, K * l, device=device)

        # Get model predictions
        mean_vec = model(x)

        # Generate proposals
        proposal = mean_vec + sigma * torch.randn_like(mean_vec)

        # Calculate gradients
        p_theta = (proposal - mean_vec) / (sigma ** 2)

        # Generate samples with BPSK modulation
        samples = torch.sign(proposal.view(N, K, l))
    return samples, p_theta

@torch.no_grad()
def objective_function(samples):
    '''
    Compute the objective function for the given samples.
    
    Parameters:
    -----------
    samples (torch.Tensor): N code families of K codes of length l.
    
    Returns:
    --------
    torch.Tensor: N objective function values.
    '''
    N, K, l = samples.shape
    samples = samples.float() # Ensure float32 for FFT

    with autocast(device_type=samples.device.type, enabled=False):
        # Compute the FFT of all samples
        fft_samples = fft(samples, dim=2)

        # Compute the correlation tensor
        corr_matrix = torch.einsum('nkl,nml->nkml', fft_samples, torch.conj(fft_samples))
        corr_matrix = ifft(corr_matrix, dim=3).real / l

    corr_matrix[:, torch.arange(K), torch.arange(K), 0] -= 1  # Remove central peak

    # Compute the objective function
    corr_power = corr_matrix ** 2
    corr_sum = torch.sum(corr_power, dim=3)
    autocorr = torch.sum(torch.diagonal(corr_sum, dim1=1, dim2=2), dim=1)
    crosscorr = torch.sum(corr_sum.view(N, -1), dim=1) - autocorr
    f_ac = autocorr / (K * l)
    # Number of pairs of codes
    K_p = K * (K - 1)
    f_cc = crosscorr / (K_p * l)
    obj = torch.max(f_ac, f_cc)

    return obj

def nes_gradient_step(model, optimizer, obj_values, log_prob, scaler):
    '''
    Perform a single gradient step using the Natural Evolution Strategy (NES) algorithm.

    Parameters:
    -----------
    model (torch.nn.Module): Code generator model.
    optimizer (torch.optim.Optimizer): Optimizer.
    obj_values (torch.Tensor): N objective function values.
    log_prob (torch.Tensor): N times ∇_θ*log(p_θ(x^i)).
    scaler (GradScaler): Scaler for mixed precision training.

    Returns:
    --------
    loss (float): Loss value (for logging).
    '''
    N = obj_values.shape[0]

    # Compute baseline value
    b = torch.mean(obj_values)

    # Compute the gradient estimate
    weights = (obj_values - b).view(N, 1)
    grad_estimate = torch.mean(weights * log_prob, dim=0)

    # Update model parameters
    optimizer.zero_grad()
    x_dummy = torch.randn(1, grad_estimate.numel(), device=grad_estimate.device)

    with autocast(device_type=grad_estimate.device.type):
        y_dummy = model(x_dummy)

    scaler.scale(y_dummy).backward(grad_estimate.view(1, -1))
    scaler.step(optimizer)
    scaler.update()

    # Compute the maximum loss (for logging)
    loss = torch.max(obj_values).item()
    return loss

def train_model(K, l, batch_size, N_epochs, device, freq_progress=100):
    '''
    Training loop for the model.

    Parameters:
    -----------
    K (int): Number of codes per code family.
    l (int): Code length.
    batch_size (int): Batch size.
    N_epochs (int): Number of epochs.
    device (str): Device to run the model on.
    freq_progress (int): Frequency of progress updates.

    Returns:
    --------
    None
    '''
    mpath = f"models/l={l}/K={K}.pt.gz"
    pltpath = f"plots/training/l={l}/K={K}.png"
    datapath = f"data/l={l}.csv"

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(mpath), exist_ok=True)
    os.makedirs(os.path.dirname(pltpath), exist_ok=True)
    os.makedirs(os.path.dirname(datapath), exist_ok=True)

    # Model initialization
    model = CodeGen(K, l).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5 if l >= 500 else 1e-4)
    scaler = GradScaler()

    # Training loop
    torch.backends.cudnn.benchmark = False
    losses = []
    epoch = 0
    # Start the timer
    start = datetime.datetime.now()
    print(f"Training model with K={K}, l={l} on {device}")
    while epoch < N_epochs:
        
        with autocast(device_type=device.type):
            samples, p_theta = get_samples(model, K, l, batch_size, device=device)
        # Protect against memory leaks
        with torch.no_grad():
            obj_values = objective_function(samples)
        loss = nes_gradient_step(model, optimizer, obj_values, p_theta, scaler)
        losses.append(loss)

        del samples, p_theta, obj_values # Clear memory

        # Output training progress every freq_progress epochs...
        # ...this also trims the cache and collects garbage
        if epoch % freq_progress == 0:
            print("--------------------------------------------------")
            print(f"EPOCH #{epoch} ({(epoch / N_epochs) * 100:.3f})\nLoss: {loss}")
            if device.type == 'cuda':
                # Clear memory
                torch.cuda.empty_cache()
                gc.collect()
                # Print memory usage
                memory_allocated = torch.cuda.memory_allocated(device)
                max_memory_allocated = torch.cuda.max_memory_allocated(device)
                print(f"Mem. %: {(memory_allocated / max_memory_allocated) * 100:.3f}")
            # Timestamp for slurm batched output
            print(f"Time elapsed: {datetime.datetime.now() - start}")
        epoch += 1

    # Compress and save the model
    with gzip.open(mpath, 'wb') as f:
        torch.save(model.state_dict(), f)
    print(f"Model saved at {mpath}")

    # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(r"$\max(f_{\text{AC}}(x^i), f_{\text{CC}}(x^i))$ vs Epochs" + f" ($\\ell$={l}, $K$={K})")
    plt.grid()
    plt.tight_layout()
    plt.savefig(pltpath, dpi=300)
    print(f"Plot saved at {pltpath}")

    # Log the final training loss in a CSV file (append to other K values)
    tools.log_training_result(K, losses, datapath=datapath)

if __name__ == "__main__":
    # Code parameters
    K_list = [3, 5, 7, 10, 13, 15, 18, 20, 25, 31]
    l_list = [63, 67, 127, 257, 511, 1023, 1031]

    # Leave possibility of adapting batch size
    initial_batch_size = 100
    N_epochs = 10000

    # Prioritize GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a list of tasks
    tasks = [(K, l) for l in l_list for K in K_list]

    # Run tasks sequentially
    for K, l in tasks:
        batch_size = initial_batch_size
        train_model(K, l, batch_size, N_epochs, device)