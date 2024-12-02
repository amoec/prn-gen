# eval.py
# Description: This file generates evaluation metrics for the model.

import numpy as np
import matplotlib.pyplot as plt
import os

def main(data1, data2, case, savepath):
    """
    Plot the evaluation metrics for the model.

    Parameters:
    -----------
    data1 (np.ndarray): Data from trained model.
    data2 (np.ndarray): Data from Mina and Gao (2022).
    case (int): Code length.
    savepath (str): Path to save the plot.

    Returns:
    --------
    None
    """
    # Plot the data
    plt.figure(figsize=(6, 6))
    plt.plot(data1[:, 0], data1[:, 1], label='Trained Model', color='blue', marker='o')
    plt.plot(data2[:, 0], data2[:, 1], label='Mina and Gao (2022)', color='red', linestyle='--', marker='x')
    plt.xlabel(r'Code family size ($K$)')
    plt.ylabel('Normalized mean squared correlation performance')
    plt.title(f'Model Evaluation for $\ell={case}$')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(savepath, dpi=600)
    print(f"Plot saved at {savepath}")

if __name__ == "__main__":
    l_list = [63, 127, 511]

    for l in l_list:
        # Load the data
        datapath1 = f"data/training/l={l}.csv"
        datapath2 = f"data/eval/l={l}.csv"

        data1 = np.loadtxt(datapath1, delimiter=',', skiprows=1)
        data2 = np.loadtxt(datapath2, delimiter=',', skiprows=1)        # Plot the data
        
        savepath = f"plots/eval/l={l}.png"
        main(data1, data2, l, savepath)
    print("Evaluation complete.")