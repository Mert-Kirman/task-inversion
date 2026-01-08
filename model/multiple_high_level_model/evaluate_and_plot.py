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
run_id = "run_1767830308.749522"
save_path = f"model/multiple_high_level_model/save/{run_id}"

# POINT TO THE PAIRED DATA FOLDER
data_path = "data/paired_trajectories_insert_place" 

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
    
    # Extract Y stats
    # Check if they are lists or tensors (depends on how train.py saved them)
    if isinstance(stats['Y_min'], list):
        y_min = torch.stack(stats['Y_min'])
        y_max = torch.stack(stats['Y_max'])
    else:
        y_min = stats['Y_min']
        y_max = stats['Y_max']
    
    # Extract Context stats
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

def load_matched_data():
    """
    Loads matched insert/place data and reconstructs the Averaged Context.
    Returns RAW (un-normalized) tensors.
    """
    print(f"Loading paired data from {data_path}...")
    
    insert_file = os.path.join(data_path, 'insert_all.npy')
    place_file = os.path.join(data_path, 'place_all.npy')
    
    if not os.path.exists(insert_file) or not os.path.exists(place_file):
        print(f"Error: Data files not found in {data_path}")
        sys.exit(1)

    # Load arrays of dicts
    insert_data = np.load(insert_file, allow_pickle=True)
    place_data = np.load(place_file, allow_pickle=True)

    # Extract Trajectories (Batch, Time, Dim)
    # We take the first 3 dims (x, y, z)
    Y1_list = [d['pose'][0][:, :3] for d in insert_data] 
    Y2_list = [d['pose'][0][:, :3] for d in place_data]

    # We trained with top matched only
    top_x_matched = min(10, len(Y1_list))
    Y1_list = Y1_list[:top_x_matched]
    Y2_list = Y2_list[:top_x_matched]
    print(f"Using top {top_x_matched} matched trajectories for evaluation.")

    Y1_raw = torch.tensor(np.stack(Y1_list), dtype=torch.float32)
    Y2_raw = torch.tensor(np.stack(Y2_list), dtype=torch.float32)
    
    # --- CONTEXT RECONSTRUCTION (AVERAGED) ---
    # C = (Insert_End_XY + Place_Start_XY) / 2
    
    insert_ends_xy = Y1_raw[:, -1, :2]
    place_starts_xy = Y2_raw[:, 0, :2]
    
    C_raw = (insert_ends_xy + place_starts_xy) / 2.0
    
    print(f"Data loaded. Found {Y1_raw.shape[0]} matched pairs.")
    return Y1_raw, Y2_raw, C_raw

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
    # 1. Load Norm Stats
    y_min, y_max, c_min, c_max = load_normalization_stats()
    
    # 2. Load Raw Matched Data
    # Y1 = Insert (Forward), Y2 = Place (Inverse)
    Y1_raw, Y2_raw, C_raw = load_matched_data()
    
    d_x = 1
    d_y1 = Y1_raw.shape[2] 
    d_y2 = Y2_raw.shape[2] 
    d_param = C_raw.shape[1]
    time_len = Y1_raw.shape[1] 
    num_demos = Y1_raw.shape[0]

    # --- NORMALIZE CONTEXT ---
    # We must normalize C using the stats from training
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
    
    # 5. Define Condition Points
    time_steps = np.linspace(0, 1, time_len)
    cond_step_indices = [300] 
    
    # 6. Plot Setup
    fig, axes = plt.subplots(num_to_plot, d_y1, figsize=(15, 4 * num_to_plot))
    if num_to_plot == 1: axes = np.expand_dims(axes, 0) 

    print(f"Evaluating indices: {indices}")

    for row_idx, traj_idx in enumerate(indices):
        
        # --- A. Prepare Ground Truth ---
        curr_y_truth_raw = Y2_raw[traj_idx].numpy() # Place Action (Inverse)
        
        # --- B. Prepare Input (Conditioning on Insert action) ---
        # The model is predicting Y2 (Place) given Y1 (Insert)
        condition_points = []
        for t_idx in cond_step_indices:
            t_val = time_steps[t_idx]
            
            # Get raw value from Y1 (Insert)
            y_val_raw = Y1_raw[traj_idx, t_idx:t_idx+1] # Shape (1, d_y1)
            
            # Normalize
            y_val_norm = normalize_data(y_val_raw, y_min, y_max)
            condition_points.append([t_val, y_val_norm])
        
        curr_context = C_normalized[traj_idx]

        # --- C. Run Inference (Inverse Mode) ---
        with torch.no_grad():
            means_norm, stds_norm = model_predict.predict_inverse(
                model, time_len, curr_context, condition_points, d_x, d_y1, d_y2
            )
            
        # --- D. Denormalize Output ---
        means_pred = denormalize_data(means_norm, y_min, y_max)
        stds_pred = stds_norm * (y_max - y_min)

        # --- E. Plotting ---
        dim_labels = ["X (Place)", "Y (Place)", "Z (Place)"]
        
        for col_idx in range(d_y1):
            ax = axes[row_idx, col_idx]
            
            # 1. Ground Truth
            ax.plot(time_steps, curr_y_truth_raw[:, col_idx], 
                    color='black', linestyle='-', linewidth=2, alpha=0.5, label='GT (Place)')
            
            # 2. Prediction
            ax.plot(time_steps, means_pred[:, col_idx].numpy(), 
                    color='blue', linestyle='--', linewidth=2, label='Pred (Place)')
            
            # 3. Uncertainty
            sigma = stds_pred[:, col_idx].numpy()
            mean_curve = means_pred[:, col_idx].numpy()
            ax.fill_between(time_steps, mean_curve - 2*sigma, mean_curve + 2*sigma, 
                            color='blue', alpha=0.1, label='Uncertainty')

            # Labels
            if row_idx == 0:
                ax.set_title(dim_labels[col_idx], fontsize=14, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f"Pair {traj_idx}", fontsize=10)

            ax.grid(True, alpha=0.3)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize='small', loc='best')

    plt.suptitle(f"Inverse Task (Place) Evaluation\nContext: Avg(Insert_End, Place_Start)", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) 
    
    save_file = f'{save_path}/eval_inverse_place.png'
    plt.savefig(save_file)
    print(f"Evaluation plots saved to {save_file}")

if __name__ == "__main__":
    plot_training_progress()
    evaluate_random_trajectories(num_samples=56)
