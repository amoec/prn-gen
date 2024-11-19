# main.py
# Description: This file contains the main training loop for the model.

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.fft import fft, ifft
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

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
    scaler (torch.amp.GradScaler): Scaler for mixed precision training.

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

    y_dummy = model(x_dummy)
    scaler.scale(y_dummy).backward(grad_estimate.view(1, -1))
    scaler.step(optimizer)
    scaler.update()

    # Compute the maximum loss (for logging)
    loss = torch.max(obj_values).item()
    return loss

# CODE PARAMETERS
K_list = [3, 5, 7, 10, 13, 15, 18, 20, 25, 31]
l_list = [511, 1023, 1031]

# TRAINING PARAMETERS
batch_size = 100
N_epochs = 10000
device = 'cuda'

for l in l_list:
    for K in K_list:
        mpath = f"models/l={l}/K={K}.pt"
        pltpath = f"plots/training/l={l}/K={K}.png"
        datapath = f"data/l={l}.csv"

        os.makedirs(os.path.dirname(mpath), exist_ok=True)
        os.makedirs(os.path.dirname(pltpath), exist_ok=True)
        os.makedirs(os.path.dirname(datapath), exist_ok=True)

        # MODEL INIT
        model = CodeGen(K, l).to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-5 if l >= 500 else 1e-4)
        torch.backends.cudnn.benchmark = True
        scaler = GradScaler()
        losses = []

        # TRAINING
        for epoch in range(N_epochs):
            samples, p_theta = get_samples(model, K, l, batch_size, device=device)
            obj_values = objective_function(samples)
            loss = nes_gradient_step(model, optimizer, obj_values, p_theta, scaler)
            losses.append(loss)
            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Loss: {loss}')
        
        # STORING
        torch.save(model.state_dict(), mpath)
        print(f"Model saved at {mpath}")

        # PLOTTING
        plt.figure(figsize=(10, 6))
        plt.plot(losses, 'b')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(r"$\max(f_{\text{AC}}(x^i), f_{\text{CC}}(x^i))$ vs Epochs" + f" ($\\ell$={l}, $K$={K})")
        plt.grid()
        plt.tight_layout()
        plt.savefig(pltpath, dpi=300)
        print(f"Plot saved at {pltpath}")

        # CSV LOGGING
        tools.log_training_result(K, losses, datapath=datapath)
