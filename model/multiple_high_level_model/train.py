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
import model.single_high_level_model.dual_enc_dec_model as dual_enc_dec_cnmp
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

    unpaired_traj = True 
    
    for i in tqdm(range(EPOCHS)):

        extra_pass = False
        if unpaired_traj:
            p = np.random.random_sample()
            if p < 0.20:
                extra_pass = True

        obs, params, mask, x_tar, y_tar_f, y_tar_i, extra_pass = dual_enc_dec_cnmp.get_training_sample(extra_pass, valid_inverses, validation_indices, demo_data, OBS_MAX, d_N, d_x, d_y1, d_y2, d_param, time_len)
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
    data_folder = f"data/processed_relative_high_level_actions/pick/round_peg_4"
    robot_state_sensor_names = ['compensated_base_force', 'compensated_base_torque', 'gripper_positions', 'joint_efforts', 'joint_positions', 'joint_velocities', 'measured_force', 'measured_torque', 'pose', 'velocity']

    # Load trajectory(sensorimotor) data
    trajectory_list = []
    for file in os.listdir(data_folder):
        if file.endswith(".npy"):
            high_level_action_dict_interpolated = np.load(os.path.join(data_folder, file), allow_pickle=True).item()
    
            modality_files_interpolated = {}
            for sensor in robot_state_sensor_names:
                if sensor not in high_level_action_dict_interpolated:
                    continue
                sensor_values = high_level_action_dict_interpolated[sensor][0]
                timestamps = high_level_action_dict_interpolated[sensor][1]
                modality_files_interpolated[sensor] = (sensor_values, timestamps)
            trajectory_list.append(modality_files_interpolated)

    selected_sensors = ['pose']
    
    # Process each trajectory and stack them
    trajectory_arrays = []
    for trajectory in trajectory_list:
        # Concatenate sensor values for this trajectory across selected sensors
        sensor_data = [trajectory[sensor][0] for sensor in selected_sensors if sensor in trajectory]
        trajectory_array = np.concatenate(sensor_data, axis=1)  # Shape: (1000, 7)
        trajectory_arrays.append(trajectory_array[:, :3])  # Keep only position (x, y, z)
    
    # Stack all trajectories to get shape (56, 1000, 3)
    Y1 = torch.tensor(np.stack(trajectory_arrays, axis=0), dtype=torch.float32)
    Y2 = Y1.detach().clone()
    
    # # Use Start Position as Context (Geometric Context)
    # # We take X and Y from the first timestamp (t=0)
    # # Shape: (56, 2)
    # C = Y1[:, 0, :2].clone() 

    # Normalization Logic
    print("Normalizing Data (Min-Max)...")
    
    # 1. Normalize Trajectories (Y)
    Y_min_vals = []
    Y_max_vals = []
    
    for dim in range(Y1.shape[2]):
        min_dim = torch.minimum(Y1[:, :, dim].min(), Y2[:, :, dim].min())
        max_dim = torch.maximum(Y1[:, :, dim].max(), Y2[:, :, dim].max())
        
        # Save for later use/inference
        Y_min_vals.append(min_dim)
        Y_max_vals.append(max_dim)
        
        denominator = max_dim - min_dim
        
        if denominator == 0:
            Y1[:, :, dim] = 0.0 
            Y2[:, :, dim] = 0.0
        else:
            Y1[:, :, dim] = (Y1[:, :, dim] - min_dim) / denominator
            Y2[:, :, dim] = (Y2[:, :, dim] - min_dim) / denominator

    C = torch.zeros((Y1.shape[0], 1))
    # # 2. Normalize Context (C)
    # C_min_val = C.min(dim=0)[0]
    # C_max_val = C.max(dim=0)[0]
    # C_denom = C_max_val - C_min_val
    
    # # Avoid div by zero in context
    # C_denom[C_denom == 0] = 1.0 
    
    # C = (C - C_min_val) / C_denom
    
    # print(f"Context Normalized. Range: [{C.min()}, {C.max()}]")

    num_demo = Y1.shape[0]
    time_len = Y1.shape[1]

    X1 = torch.linspace(0, 1, time_len).repeat(num_demo, 1).reshape(num_demo, -1, 1)
    X2 = torch.linspace(0, 1, time_len).repeat(num_demo, 1).reshape(num_demo, -1, 1)

    valid_inverses = [True for _ in range(num_demo)]  # All trajectories have valid inverses in this setup

    d_x = 1
    d_param = C.shape[1]
    d_y1 = Y1.shape[2]
    d_y2 = Y2.shape[2]

    OBS_MAX = 10
    d_N = num_demo

    all_indices = set(range(num_demo))
    validation_indices = [i for i in range(0, num_demo, 4)] # Indices of trajectories used for validation
    training_indices = list(all_indices - set(validation_indices)) # Indices of trajectories used for training

    demo_data = [X1, X2, Y1, Y2, C]

    save_folder = f"model/multiple_high_level_model/save"
    run_id = time.time()
    os.makedirs(f'{save_folder}/run_{run_id}', exist_ok=True)

    # Save Normalization Constants for Inference
    print("Saving Normalization Stats...")
    np.save(f'{save_folder}/run_{run_id}/normalization_stats.npy', {
        'Y_min': Y_min_vals, 'Y_max': Y_max_vals,
        # 'C_min': C_min_val, 'C_max': C_max_val
    })

    errors = []
    losses = []
    errors_with_latent = []

    EPOCHS = 60_001
    learning_rate = 3e-4
    model = dual_enc_dec_cnmp.DualEncoderDecoder(d_x, d_y1, d_y2, d_param)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 if epoch < 40_000 else 5e-1)

    training_errors, validation_errors, losses = train(model, optimizer, scheduler, EPOCHS, unpaired_traj=True)
