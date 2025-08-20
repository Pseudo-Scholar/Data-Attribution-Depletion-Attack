import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.func import functional_call, grad, vmap
from torch.nn.utils import parameters_to_vector
import time  # Ensure time module is imported

# -------------------- Compute Attack Success Rate (for CIFAR dataset) --------------------
def compute_attack_success_rate(base_dir, original_shapley_file, poisoned_shapley_file, mapping_file):
    # Adjust paths to the results2 folder
    original_shapley_path = os.path.join(base_dir, 'results2', original_shapley_file)
    poisoned_shapley_path = os.path.join(base_dir, 'results2', poisoned_shapley_file)
    mapping_path = os.path.join(base_dir, 'results2', mapping_file)

    # Read original Shapley values (adapted for CIFAR column name: sample_index)
    original_shapley = {}
    with open(original_shapley_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_shapley[int(row['sample_index'])] = float(row['shapley_value'])
    
    # Read poisoned Shapley values (adapted for CIFAR column name: sample_index)
    poisoned_shapley = {}
    with open(poisoned_shapley_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            poisoned_shapley[int(row['sample_index'])] = float(row['shapley_value'])
    
    # Read poisoned index mapping (adapted for CIFAR column name: original_index)
    poisoned_indices = []
    with open(mapping_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            poisoned_indices.append(int(row['original_index']))

    # Initialize statistics
    success_count = 0
    total_poisoned = 0
    
    # Variables for Forged Gain Success Rate
    valid_orig_sum = 0.0
    valid_pois_sum = 0.0
    valid_pairs_count = 0
    
    # Variables for Loss Rise Rate
    good_samples_discarded = 0  # orig>0 and pois<0
    bad_samples_retained = 0    # orig<0 and pois>0

    # Iterate through all poisoned sample indices
    for idx in poisoned_indices:
        if idx in original_shapley and idx in poisoned_shapley:
            orig = original_shapley[idx]
            pois = poisoned_shapley[idx]
            total_poisoned += 1
            
            # Compute attack success rate
            if pois < orig or (orig < 0 and pois > 0):
                success_count += 1
            
            # Compute Forged Gain Success Rate (excluding cases where orig < 0 and pois > 0)
            if not (orig < 0 and pois > 0):
                valid_orig_sum += orig
                valid_pois_sum += pois
                valid_pairs_count += 1
            
            # Compute Loss Rise Rate
            if orig > 0 and pois < 0:
                good_samples_discarded += 1
            elif orig < 0 and pois > 0:
                bad_samples_retained += 1

    # Calculate and print Attack Success Rate
    if total_poisoned > 0:
        success_rate = success_count / total_poisoned
        print(f"🎯 Attack Success Rate: {success_rate:.2%} ({success_count}/{total_poisoned})")
    else:
        print("⚠️ No poisoned data indices found for evaluating attack success rate.")
        success_rate = 0

    # Calculate and print Forged Gain Success Rate
    if valid_pairs_count > 0:
        forged_gain_success = 1.0 if valid_orig_sum > valid_pois_sum else 0.0
        print(f"💰 Forged Gain Success Rate: {forged_gain_success:.2%} (Original Sum: {valid_orig_sum:.4f}, Poisoned Sum: {valid_pois_sum:.4f})")
    else:
        print("⚠️ No valid pairs found for evaluating forged gain success rate.")
        forged_gain_success = 0

    # Calculate and print Loss Rise Rate
    loss_rise_count = good_samples_discarded + bad_samples_retained
    loss_rise_rate = 1.0 if loss_rise_count > 0 else 0.0
    print(f"📈 Loss Rise Rate: {loss_rise_rate:.2%} (Good samples discarded: {good_samples_discarded}, Bad samples retained: {bad_samples_retained})")
    
    return {
        'attack_success_rate': success_rate,
        'forged_gain_success_rate': forged_gain_success,
        'loss_rise_rate': loss_rise_rate
    }

# -------------------- Main Function -------- #
def main():
    base_dir = "./xie/FNN_Shapley"
    # Change to the filenames for the CIFAR dataset
    original_shapley_file = "shapley_original_0.1.csv"
    poisoned_shapley_file = "shapley_backdoor_0.1.csv"
    mapping_file = "backdoor_original_index_mapping_0.1.csv"
    
    compute_attack_success_rate(base_dir, original_shapley_file, poisoned_shapley_file, mapping_file)
    
if __name__ == "__main__":
    main()