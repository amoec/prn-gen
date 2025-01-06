# utils.py
# Description: This file contains utility functions for model/evaluation.

import csv
import matplotlib.pyplot as plt
import os
from typing import *

def log_training_result(K, losses, datapath):
    '''
    Log the training result to a CSV file.

    Parameters:
    -----------
    K (int): Number of codes per code family.
    losses (list): List of losses during training.
    datapath (str): Path to the CSV file to save the data.
    '''
    # Prepare the new row
    new_row = [int(K), losses[-1]]

    # Check if the file exists
    file_exists = os.path.isfile(datapath)

    # Open the file in append mode ('a')
    with open(datapath, mode='a', newline='') as f:
        writer = csv.writer(f)

        # Write header only if the file doesn't exist
        if not file_exists:
            writer.writerow(['K', 'f_obj'])

        # Append the new row
        writer.writerow(new_row)

    print(f"Data saved at {datapath}")
