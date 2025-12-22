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
import numpy as np
import model.validate_model as validate_model
import model.multiple_high_level_model.dual_enc_dec_model as dual_enc_dec_cnmp
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import time
import wandb


# ================= HELPER FUNCTIONS =================
def load_and_process_data(data_folder, selected_sensors=['pose']):
    robot_state_sensor_names = ['compensated_base_force', 'compensated_base_torque', 'gripper_positions', 'joint_efforts', 'joint_positions', 'joint_velocities', 'measured_force', 'measured_torque', 'pose', 'velocity']

    trajectory_list = []
    # Load all files
    files = sorted([f for f in os.listdir(data_folder) if f.endswith(".npy")])
    
    for file in files:
        path = os.path.join(data_folder, file)
        high_level_action_dict_interpolated = np.load(path, allow_pickle=True).item()
        modality_files_interpolated = {}
        for sensor in robot_state_sensor_names:
            if sensor not in high_level_action_dict_interpolated: continue
            sensor_values = high_level_action_dict_interpolated[sensor][0]
            timestamps = high_level_action_dict_interpolated[sensor][1]
            modality_files_interpolated[sensor] = (sensor_values, timestamps)
        trajectory_list.append(modality_files_interpolated)

    # Process and stack
    trajectory_arrays = []
    for trajectory in trajectory_list:
        sensor_data = [trajectory[sensor][0] for sensor in selected_sensors if sensor in trajectory]
        trajectory_array = np.concatenate(sensor_data, axis=1)
        trajectory_arrays.append(trajectory_array[:, :3]) # Keep X,Y,Z
    
    Y1 = torch.tensor(np.stack(trajectory_arrays, axis=0), dtype=torch.float32)
    Y2 = Y1.detach().clone()
    
    # Geometric Context: Start X, Y
    C = Y1[:, 0, :2].clone() 

    # Normalization
    Y_min_vals, Y_max_vals = [], []
    for dim in range(Y1.shape[2]):
        min_dim = torch.minimum(Y1[:, :, dim].min(), Y2[:, :, dim].min())
        max_dim = torch.maximum(Y1[:, :, dim].max(), Y2[:, :, dim].max())
        Y_min_vals.append(min_dim)
        Y_max_vals.append(max_dim)
        denom = max_dim - min_dim
        denom[denom == 0] = 1.0
        Y1[:, :, dim] = (Y1[:, :, dim] - min_dim) / denom
        Y2[:, :, dim] = (Y2[:, :, dim] - min_dim) / denom

    C_min = C.min(dim=0)[0]
    C_max = C.max(dim=0)[0]
    C_denom = C_max - C_min
    C_denom[C_denom == 0] = 1.0
    C = (C - C_min) / C_denom

    return Y1, Y2, C, Y_min_vals, Y_max_vals, C_min, C_max

def train():
    # 1. Initialize WandB
    # We don't pass config here if we are running a sweep; wandb handles it.
    # If running standalone, we can set default config.
    wandb.init(project="cnmp-pick-task", config={
        "learning_rate": 3e-4,
        "weight_decay": 1e-5,
        "batch_size": 1,
        "obs_max": 10,
        "epochs": 60001,
        "dropout": 0.0,
        "scheduler_step": 40000
    })
    
    config = wandb.config # Access hyperparameters

    # 2. Data Loading (Cached or Reloaded)
    data_folder = "data/processed_relative_high_level_actions/pick/round_peg_4"
    Y1, Y2, C, _, _, _, _ = load_and_process_data(data_folder)
    
    num_demo = Y1.shape[0]
    time_len = Y1.shape[1]
    
    # Prepare Tensors
    X1 = torch.linspace(0, 1, time_len).repeat(num_demo, 1).reshape(num_demo, -1, 1)
    X2 = torch.linspace(0, 1, time_len).repeat(num_demo, 1).reshape(num_demo, -1, 1)
    
    # Validation Split
    valid_inverses = [True] * num_demo
    validation_indices = [i for i in range(0, num_demo, 4)]
    training_indices = list(set(range(num_demo)) - set(validation_indices))
    
    demo_data = [X1, X2, Y1, Y2, C]
    
    # 3. Model Setup
    d_x = 1
    d_param = C.shape[1]
    d_y1 = Y1.shape[2]
    d_y2 = Y2.shape[2]
    
    # Pass dropout from config
    model = dual_enc_dec_cnmp.DualEncoderDecoder(d_x, d_y1, d_y2, d_param, dropout_p=[config.dropout, config.dropout])
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 if epoch < config.scheduler_step else 0.5)

    # 4. Training Loop
    unpaired_traj = True
    
    # Progress bar
    pbar = tqdm(range(config.epochs))
    
    for i in pbar:
        extra_pass = False
        if unpaired_traj and np.random.random() < 0.20:
            extra_pass = True

        # Use config.obs_max
        obs, params, mask, x_tar, y_tar_f, y_tar_i, extra_pass = dual_enc_dec_cnmp.get_training_sample(
            extra_pass, valid_inverses, validation_indices, demo_data, 
            config.obs_max, num_demo, d_x, d_y1, d_y2, d_param, time_len
        )
        
        optimizer.zero_grad()
        output, L_F, L_I, extra_pass = model(obs, params, mask, x_tar, extra_pass)
        
        # Loss
        loss_val = dual_enc_dec_cnmp.loss(output, y_tar_f, y_tar_i, d_y1, d_y2, d_param, L_F.squeeze(1), L_I.squeeze(1), extra_pass)
        loss_val.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        # Logging & Validation
        if i % 1000 == 0:
            train_err = validate_model.val_only_extra(model, training_indices, i, demo_data, d_x, d_y1, d_y2, time_len=time_len)
            val_err = validate_model.val_only_extra(model, validation_indices, i, demo_data, d_x, d_y1, d_y2, time_len=time_len)
            
            # WANDB LOGGING
            wandb.log({
                "train_loss": loss_val.item(),
                "train_mse": train_err,
                "val_mse": val_err
            })
            
            pbar.set_description(f"Loss: {loss_val.item():.4f} | Val MSE: {val_err:.4f}")

    # Finish run
    wandb.finish()

if __name__ == "__main__":
    train()