import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math
import importlib
import model.validate_model as validate_model
import model.multiple_high_level_model.dual_enc_dec_model as dual_enc_dec_cnmp
import model.utils as utils
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import time

def train(model, optimizer, scheduler, EPOCHS, unpaired_traj=True):

    os.makedirs(f'model/multiple_high_level_model/logs/run_{run_id}/', exist_ok=True)
    sys.stdout = open(f'model/multiple_high_level_model/logs/run_{run_id}/train_log.txt', 'w')

    training_errors = []
    validation_errors = []
    losses = []

    # Batch Size = 1 was found to be best for this dataset size
    BATCH_SIZE = 1
    
    for i in tqdm(range(EPOCHS)):

        extra_pass = False
        if unpaired_traj:
            p = np.random.random_sample()
            if p < 0.20:
                extra_pass = True

        # Note: We pass batch_size explicitly here
        obs, params, mask, x_tar, y_tar_f, y_tar_i, extra_pass = dual_enc_dec_cnmp.get_training_sample(
            extra_pass, valid_inverses, validation_indices, demo_data, 
            OBS_MAX, d_N, d_x, d_y1, d_y2, d_param, time_len, 
            batch_size=BATCH_SIZE
        )
        
        optimizer.zero_grad()
        output, L_F, L_I, extra_pass = model(obs, params, mask, x_tar, extra_pass)
        
        loss = dual_enc_dec_cnmp.loss(output, y_tar_f, y_tar_i, d_y1, d_y2, d_param, L_F.squeeze(1), L_I.squeeze(1), extra_pass)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()
        scheduler.step()

        if i > 0 and i % 1000 == 0:
            epoch_train_error = validate_model.val_only_extra(model, training_indices, i, demo_data, d_x, d_y1, d_y2, time_len=time_len)
            training_errors.append(epoch_train_error)

            epoch_val_error = validate_model.val_only_extra(model, validation_indices, i, demo_data, d_x, d_y1, d_y2, time_len=time_len)
            validation_errors.append(epoch_val_error)
            
            losses.append(loss.item())

            # Save errors and losses
            np.save(f'{save_folder}/run_{run_id}/training_errors_mse.npy', np.array(training_errors))
            np.save(f'{save_folder}/run_{run_id}/validation_errors_mse.npy', np.array(validation_errors))
            np.save(f'{save_folder}/run_{run_id}/losses_log_prob.npy', np.array(losses))

            if min(validation_errors) == validation_errors[-1]:
                # Save model
                tqdm.write(f"Run ID: {run_id}, Saved model epoch {i}, Train loss: {loss.item():6f}, Validation error: {epoch_val_error:6f}")
                torch.save(model.state_dict(), f'{save_folder}/run_{run_id}/perfectly_paired.pth')

    return training_errors, validation_errors, losses


if __name__ == "__main__":
    # --- LOAD MATCHED DATA ---
    data_folder = "data/paired_trajectories_insert_place"
    insert_path = os.path.join(data_folder, 'insert_all.npy')
    place_path = os.path.join(data_folder, 'place_all.npy')

    if not os.path.exists(insert_path) or not os.path.exists(place_path):
        print(f"Error: Could not find matched files in {data_folder}")
        sys.exit(1)

    print(f"Loading paired data from {data_folder}...")
    # These load as arrays of dictionaries (saved by match_and_stack_trajectories.py)
    insert_data = np.load(insert_path, allow_pickle=True)
    place_data = np.load(place_path, allow_pickle=True)

    # --- EXTRACT TRAJECTORIES (X, Y, Z) ---
    # Extract 'pose' values from the dictionary and keep first 3 dims (x, y, z)
    # The structure inside npy is: array([dict, dict, ...])
    # dict['pose'][0] is the value array (Time, Dims)
    
    Y1_list = [d['pose'][0][:, :3] for d in insert_data] # Forward (Insert)
    Y2_list = [d['pose'][0][:, :3] for d in place_data]  # Inverse (Place)

    # Train with top x matched trajectories only
    top_x_matched = min(200, len(Y1_list))
    Y1_list = Y1_list[:top_x_matched]
    Y2_list = Y2_list[:top_x_matched]
    print(f"Using top {top_x_matched} matched trajectories for training.")

    # Stack into tensors: (Batch, Time, Dims)
    Y1 = torch.tensor(np.stack(Y1_list), dtype=torch.float32)
    Y2 = torch.tensor(np.stack(Y2_list), dtype=torch.float32)
    
    print(f"Data Loaded. Y1 Shape: {Y1.shape}, Y2 Shape: {Y2.shape}")

    # --- CREATE CONTEXT (AVERAGED GEOMETRY) ---
    # Logic: C = (Insert_End_XY + Place_Start_XY) / 2
    
    insert_ends_xy = Y1[:, -1, :2]  # (N, 2)
    place_starts_xy = Y2[:, 0, :2]  # (N, 2)
    
    C = (insert_ends_xy + place_starts_xy) / 2.0
    
    # Clone to detach from graph if needed (though no graph yet)
    C = C.clone()
    print(f"Context Created. Shape: {C.shape}")

    # --- NORMALIZATION (Min-Max) ---
    print("Normalizing Data (Min-Max)...")
    
    Y_min_vals = []
    Y_max_vals = []
    
    # Normalize Y1 and Y2 using combined min/max to preserve scale
    for dim in range(Y1.shape[2]):
        min_dim = torch.minimum(Y1[:, :, dim].min(), Y2[:, :, dim].min())
        max_dim = torch.maximum(Y1[:, :, dim].max(), Y2[:, :, dim].max())
        
        Y_min_vals.append(min_dim)
        Y_max_vals.append(max_dim)
        
        denominator = max_dim - min_dim
        
        if denominator == 0:
            Y1[:, :, dim] = 0.0 
            Y2[:, :, dim] = 0.0
        else:
            Y1[:, :, dim] = (Y1[:, :, dim] - min_dim) / denominator
            Y2[:, :, dim] = (Y2[:, :, dim] - min_dim) / denominator

    # Normalize Context (C)
    C_min_val = C.min(dim=0)[0]
    C_max_val = C.max(dim=0)[0]
    C_denom = C_max_val - C_min_val
    
    C_denom[C_denom == 0] = 1.0 
    
    C = (C - C_min_val) / C_denom
    
    print(f"Context Normalized. Range: [{C.min()}, {C.max()}]")

    # --- SETUP TRAINING VARIABLES ---
    num_demo = Y1.shape[0]
    time_len = Y1.shape[1]

    # Create Time inputs (X)
    X1 = torch.linspace(0, 1, time_len).repeat(num_demo, 1).reshape(num_demo, -1, 1)
    X2 = torch.linspace(0, 1, time_len).repeat(num_demo, 1).reshape(num_demo, -1, 1)

    valid_inverses = [True for _ in range(num_demo)]

    d_x = 1
    d_param = C.shape[1] # Should be 2 (avg_x, avg_y)
    d_y1 = Y1.shape[2]   # 3 (x,y,z)
    d_y2 = Y2.shape[2]   # 3 (x,y,z)

    OBS_MAX = 10
    d_N = num_demo

    # Split Train/Val
    all_indices = set(range(num_demo))
    # Using every 4th trajectory for validation
    validation_indices = [i for i in range(0, num_demo, 4)]
    print(f"Validation Indices: {validation_indices}")
    training_indices = list(all_indices - set(validation_indices))

    demo_data = [X1, X2, Y1, Y2, C]

    save_folder = f"model/multiple_high_level_model/save"
    run_id = time.time()
    os.makedirs(f'{save_folder}/run_{run_id}', exist_ok=True)

    # Save Normalization Constants for Inference
    print("Saving Normalization Stats...")
    np.save(f'{save_folder}/run_{run_id}/normalization_stats.npy', {
        'Y_min': Y_min_vals, 'Y_max': Y_max_vals,
        'C_min': C_min_val, 'C_max': C_max_val
    })

    EPOCHS = 60_001
    learning_rate = 3e-4
    model = dual_enc_dec_cnmp.DualEncoderDecoder(d_x, d_y1, d_y2, d_param, dropout_p=[0.0, 0.0])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 if epoch < 40_000 else 5e-1)

    training_errors, validation_errors, losses = train(model, optimizer, scheduler, EPOCHS, unpaired_traj=True)
