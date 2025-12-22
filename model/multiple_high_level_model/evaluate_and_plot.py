import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

# Adjust path to find model modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import model.multiple_high_level_model.dual_enc_dec_model as dual_enc_dec_cnmp
import model.model_predict as model_predict

# ================= CONFIGURATION =================
run_id = "run_1765978334.9419122"  # Update this to your latest run ID
save_path = f"model/multiple_high_level_model/save/{run_id}"
data_path = "data/processed_relative_high_level_actions/pick/round_peg_4"
model_name = "perfectly_paired.pth"

# Dimensions used in training
SELECTED_SENSORS = ['pose'] 
# =================================================

def load_normalization_stats():
    """Loads min/max values used for normalization during training."""
    stats_path = os.path.join(save_path, 'normalization_stats.npy')
    if not os.path.exists(stats_path):
        print(f"Error: Normalization stats not found at {stats_path}")
        sys.exit(1)
    
    stats = np.load(stats_path, allow_pickle=True).item()
    # stats['Y_min'] is a list of tensors, convert to single tensor for easier math
    y_min = torch.stack(stats['Y_min'])
    y_max = torch.stack(stats['Y_max'])
    
    # Check if context stats exist
    c_min = stats.get('C_min', None)
    c_max = stats.get('C_max', None)
    
    return y_min, y_max, c_min, c_max

def normalize_data(tensor, min_val, max_val):
    """Min-Max normalization to [0, 1]."""
    denominator = max_val - min_val
    denominator[denominator == 0] = 1.0
    return (tensor - min_val) / denominator

def denormalize_data(tensor, min_val, max_val):
    """Reverts [0, 1] data back to original scale."""
    denominator = max_val - min_val
    return tensor * denominator + min_val

def load_all_data():
    """
    Loads all .npy files. Returns RAW (un-normalized) data for plotting
    and constructs the Context C.
    """
    print(f"Loading data from {data_path}...")
    
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} does not exist.")
        sys.exit(1)

    trajectory_list = []
    file_names = []

    for file in sorted(os.listdir(data_path)):
        if file.endswith(".npy"):
            try:
                full_path = os.path.join(data_path, file)
                data_dict = np.load(full_path, allow_pickle=True).item()
                pose_data = data_dict['pose'][0] 
                trajectory_list.append(pose_data[:, :3]) # Keep X, Y, Z
                file_names.append(file)
            except Exception as e:
                print(f"Skipping {file}: {e}")

    if not trajectory_list:
        print("No valid trajectories found.")
        sys.exit(1)

    # Stack into Tensor (Batch, Time, Dim)
    # Shape: (56, 1000, 3)
    Y_raw = torch.tensor(np.stack(trajectory_list, axis=0), dtype=torch.float32)
    Y2_raw = Y_raw.detach().clone()
    
    # --- CONTEXT SETUP ---
    
    # Geometric Context
    C = Y_raw[:, 0, :2].clone() # Take Start X, Start Y
    
    return Y_raw, Y2_raw, C, file_names

def plot_training_progress():
    """Plots loss and error curves if they exist."""
    try:
        train_err = np.load(f'{save_path}/training_errors_mse.npy')
        val_err = np.load(f'{save_path}/validation_errors_mse.npy')
        losses = np.load(f'{save_path}/losses_log_prob.npy')

        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(losses, label='Training Loss', color='orange', alpha=0.7)
        plt.title('Log Probability Loss')
        plt.xlabel('Step')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(train_err, label='Train MSE', color='blue')
        plt.plot(val_err, label='Val MSE', color='red', linestyle='--')
        plt.title('Reconstruction Error (MSE)')
        plt.xlabel('Epoch (x1000)')
        plt.ylabel('MSE')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(train_err, label='Train MSE', color='blue')
        plt.plot(val_err, label='Val MSE', color='red', linestyle='--')
        plt.title('MSE (Log Scale)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3, which="both")

        plt.tight_layout()
        plot_save = f'{save_path}/training_progress_multi.png'
        plt.savefig(plot_save)
        print(f"Training progress saved to {plot_save}")
        plt.close()
    except FileNotFoundError:
        print("Training logs not found, skipping progress plot.")

def evaluate_random_trajectories(num_samples=3):
    # Load Norm Stats
    y_min, y_max, c_min, c_max = load_normalization_stats()
    
    # 2. Load Raw Data
    Y1_raw, Y2_raw, C_raw, file_names = load_all_data()
    
    d_x = 1
    d_y1 = Y1_raw.shape[2] 
    d_y2 = Y2_raw.shape[2] 
    d_param = C_raw.shape[1]
    time_len = Y1_raw.shape[1] 
    num_demos = Y1_raw.shape[0]

    # --- NORMALIZE CONTEXT ---
    C_normalized = C_raw.clone()
    if c_min is not None and c_max is not None:
        C_normalized = normalize_data(C_raw, c_min, c_max)

    # 3. Load Model
    model = dual_enc_dec_cnmp.DualEncoderDecoder(d_x, d_y1, d_y2, d_param)
    model_path = os.path.join(save_path, model_name)
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print(f"Loading model state from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # 4. Select Random Indices
    num_to_plot = min(num_samples, num_demos)
    random.seed(42) 
    indices = random.sample(range(num_demos), num_to_plot)
    
    # 5. Define Condition Points (Start and Middle)
    time_steps = np.linspace(0, 1, time_len)
    cond_step_indices = [0, time_len // 2] # Index 0 and 500
    
    # 6. Plot Setup
    fig, axes = plt.subplots(num_to_plot, d_y1, figsize=(15, 4 * num_to_plot))
    if num_to_plot == 1: axes = np.expand_dims(axes, 0) 

    print(f"Evaluating indices: {indices}")

    for row_idx, traj_idx in enumerate(indices):
        
        # --- A. Prepare Input for this Trajectory ---
        curr_y_truth_raw = Y2_raw[traj_idx].numpy() # Ground truth (Raw Meters)
        curr_file_name = file_names[traj_idx]
        
        # Create Condition Points (NORMALIZED)
        # The model expects [0, 1] inputs, so we must normalize the raw points 
        # using the saved stats before feeding them in.
        condition_points = []
        for t_idx in cond_step_indices:
            t_val = time_steps[t_idx]
            
            # Get raw value
            y_val_raw = Y1_raw[traj_idx, t_idx:t_idx+1] # Shape (1, d_y1)
            
            # NORMALIZE IT
            y_val_norm = normalize_data(y_val_raw, y_min, y_max)
            
            condition_points.append([t_val, y_val_norm])
        
        # Expand Context
        curr_context = C_normalized[traj_idx]

        # --- B. Run Inference (In Normalized Space) ---
        with torch.no_grad():
            means_norm, stds_norm = model_predict.predict_inverse(
                model, time_len, curr_context, condition_points, d_x, d_y1, d_y2
            )
            
        # --- C. Denormalize Output (Back to Meters) ---
        means_pred = denormalize_data(means_norm, y_min, y_max)
        
        # Stds must be scaled by the range (max-min)
        stds_pred = stds_norm * (y_max - y_min)

        # --- D. Plotting (In Physical Space) ---
        dim_labels = ["X (Position)", "Y (Position)", "Z (Position)"]
        
        for col_idx in range(d_y1):
            ax = axes[row_idx, col_idx]
            
            # 1. Ground Truth (Raw)
            ax.plot(time_steps, curr_y_truth_raw[:, col_idx], 
                    color='black', linestyle='-', linewidth=2, alpha=0.5, label='Ground Truth')
            
            # 2. Prediction (Denormalized)
            ax.plot(time_steps, means_pred[:, col_idx].numpy(), 
                    color='blue', linestyle='--', linewidth=2, label='Prediction')
            
            # 3. Uncertainty
            sigma = stds_pred[:, col_idx].numpy()
            mean_curve = means_pred[:, col_idx].numpy()
            ax.fill_between(time_steps, mean_curve - 2*sigma, mean_curve + 2*sigma, 
                            color='blue', alpha=0.1, label='Uncertainty (2$\sigma$)')
            
            # 4. Condition Points (Plotting the RAW values used to generate cond points)
            for t_idx in cond_step_indices:
                t_c = time_steps[t_idx]
                val_c = Y1_raw[traj_idx, t_idx, col_idx] # Plot raw ground truth point
                ax.scatter(t_c, val_c, color='red', s=60, zorder=5, edgecolors='white', linewidth=1.5)

            # Labels
            if row_idx == 0:
                ax.set_title(dim_labels[col_idx], fontsize=14, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f"Traj {traj_idx}\n({curr_file_name[:10]}...)", fontsize=10)

            ax.grid(True, alpha=0.3)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize='small', loc='best')

    plt.suptitle(f"Multi-Trajectory Evaluation (Run {run_id})\nConditioned at t=0.0 and t=0.5 (Normalized Model)", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) 
    
    save_file = f'{save_path}/eval_multi_traj_results_normalized.png'
    plt.savefig(save_file)
    print(f"Evaluation plots saved to {save_file}")
    # plt.show()

if __name__ == "__main__":
    plot_training_progress()
    evaluate_random_trajectories(num_samples=56)