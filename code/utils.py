# utils.py
# Description: This file contains utility functions for model/evaluation.

import csv
import os
import numpy as np
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

class Gold:
    """
    Gold Code generator.
    """
    def __init__(self, IC: np.array, taps: np.array) -> None:
        """
        Parameters:
        -----------
        IC (array_like): Initial conditions.
        taps (array_like): Taps locations.
        """
        self.LFSR = IC.astype(int)
        self.tap_idx = taps-1

    def tick(self):
        """
        Update the LFSR.
        """
        # XOR the taps
        xor_taps = np.bitwise_xor.reduce(self.LFSR[self.tap_idx])

        # Shift the LFSR
        self.LFSR[1:] = self.LFSR[:-1]

        # Update the first element
        self.LFSR[0] = xor_taps

class PRN_GC():
    """
    Pseudo-Random Noise (PRN) generator.
    """
    def __init__(self, num: int, kind: str='binary') -> None:
        """
        Parameters:
        -----------
        num (int): PRN number.
        kind (str): Type of PRN signal.
        """
        self.G1 = Gold(
            np.ones(10), # Initial conditions
            np.array([3, 10]) # Taps
            )
        
        self.G2 = Gold(
            np.ones(10), # Initial conditions
            np.array([2, 3, 6, 8, 9, 10]) # Taps
            )
        
        # PRN tap definitions
        PRN_DATABASE = {
            1: np.array([2, 6]),
            2: np.array([3, 7]),
            3: np.array([4, 8]),
            4: np.array([5, 9]),
            5: np.array([1, 9]),
            6: np.array([2, 10]),
            7: np.array([1, 8]),
            8: np.array([2, 9]),
            9: np.array([3, 10]),
            10: np.array([2, 3]),
            11: np.array([3, 4]),
            12: np.array([5, 6]),
            13: np.array([6, 7]),
            14: np.array([7, 8]),
            15: np.array([8, 9]),
            16: np.array([9, 10]),
            17: np.array([1, 4]),
            18: np.array([2, 5]),
            19: np.array([3, 6]),
            20: np.array([4, 7]),
            21: np.array([5, 8]),
            22: np.array([6, 9]),
            23: np.array([1, 3]),
            24: np.array([4, 6]),
            25: np.array([5, 7]),
            26: np.array([6, 8]),
            27: np.array([7, 9]),
            28: np.array([8, 10]),
            29: np.array([1, 6]),
            30: np.array([2, 7]),
            31: np.array([3, 8]),
            32: np.array([4, 9])
        }
        
        self.taps = PRN_DATABASE[num]-1 # Convert to 0-indexed

        if kind not in ['bin', 'bpsk', 'signal']:
            raise ValueError("Invalid kind. Choose from 'binary', 'bpsk', 'signal'.")
        else:
            self.kind = kind
    
    def __binary(self, N_c: int) -> np.array:
        """
        Generate binary PRN sequence. 
        Internal function, use generate() instead.

        Parameters:
        -----------
        N_c (int): Number of chips.

        Returns:
        --------
        PRN_bin (array_like): Binary PRN sequence.
        """
        PRN_bin = np.zeros(N_c)

        for i in range(N_c):
            PRN_bin[i] = np.bitwise_xor(self.G1.LFSR[9], 
                                        np.bitwise_xor.reduce(self.G2.LFSR[self.taps]))
            self.G1.tick() # Update GC LFSRs
            self.G2.tick()

        return PRN_bin
    
    def __bpsk(self, PRN_bin: np.array) -> np.array:
        """
        Convert binary PRN sequence to BPSK.
        Internal function, use generate() instead.

        Parameters:
        -----------
        PRN_bin (array_like): Binary PRN sequence.

        Returns:
        --------
        PRN_bpsk (array_like): BPSK PRN sequence.
        """
        return 1 - 2*PRN_bin
    
    def __signal(self, PRN_bpsk: np.array, t: np.array, f_c: float) -> np.array:
        """
        Map BPSK PRN sequence to time vector.
        Internal function, use generate() instead.

        Parameters:
        -----------
        PRN_bpsk (array_like): BPSK PRN sequence.
        t (array_like): Time vector.
        f_c (float): Chip frequency.

        Returns:
        --------
        PRN_signal (array_like): PRN signal.
        """
        T_c = 1/f_c
        f_s = 1/(t[1]-t[0])
        PRN_signal = np.zeros(len(t))

        for i in range(len(PRN_bpsk)):
            # Each chip in the BPSK sequence is T_c seconds long
            PRN_signal[np.ceil(i*T_c*f_s).astype(int):
                        np.ceil((i+1)*T_c*f_s).astype(int)] = PRN_bpsk[i]
        
        return PRN_signal
        
    def generate(self, t: Union[None, np.array]=None, N: Union[None, int]=1023,
                 f_c: Union[None, float]=1.023e6) -> np.array:
        """
        Generate PRN signal.

        Parameters:
        -----------
        t (array_like): Time vector.
        N (int): Number of chips.
        f_c (float): Chip frequency.

        Returns:
        --------
        PRN_signal (array_like): PRN sequence/signal.
        """
        PRN_bin = self.__binary(N)
        if self.kind == 'binary':
            return PRN_bin
        
        PRN_bpsk = self.__bpsk(PRN_bin)
        if self.kind == 'bpsk':
            return PRN_bpsk
        
        PRN_signal = self.__signal(PRN_bpsk, t, f_c)
        return PRN_signal