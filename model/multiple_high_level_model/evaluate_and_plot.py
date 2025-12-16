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
run_id = "run_1765922383.454245" 

save_path = f"model/multiple_high_level_model/save/{run_id}"
data_path = "data/processed_relative_high_level_actions/pick/round_peg_4"
model_name = "perfectly_paired.pth"

# Dimensions used in training
SELECTED_SENSORS = ['pose'] 
# =================================================

def load_all_data():
    """
    Loads all .npy files from the data folder and stacks them.
    Matches the logic in train.py (Pose X, Y, Z only).
    """
    print(f"Loading data from {data_path}...")
    
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} does not exist.")
        sys.exit(1)

    trajectory_list = []
    file_names = []

    # 1. Load all files
    for file in sorted(os.listdir(data_path)):
        if file.endswith(".npy"):
            try:
                full_path = os.path.join(data_path, file)
                data_dict = np.load(full_path, allow_pickle=True).item()
                
                # Extract pose
                # structure: data_dict['pose'][0] is values, [1] is timestamps
                pose_data = data_dict['pose'][0] 
                
                # Keep only X, Y, Z (first 3 columns)
                trajectory_list.append(pose_data[:, :3])
                file_names.append(file)
            except Exception as e:
                print(f"Skipping {file}: {e}")

    if not trajectory_list:
        print("No valid trajectories found.")
        sys.exit(1)

    # 2. Stack into Tensor (Batch, Time, Dim)
    # Shape: (56, 1000, 3)
    Y1 = torch.tensor(np.stack(trajectory_list, axis=0), dtype=torch.float32)
    Y2 = Y1.detach().clone()
    
    # 3. Context (Dummy zeros, matching train.py)
    C = torch.zeros((Y1.shape[0], 1))
    
    print(f"Loaded {Y1.shape[0]} trajectories with shape {Y1.shape}")
    return Y1, Y2, C, file_names

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
        
        # Zoomed in version of MSE if it dropped significantly
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
        # plt.show()
        plt.close()
    except FileNotFoundError:
        print("Training logs not found, skipping progress plot.")

def evaluate_random_trajectories(num_samples=3):
    # 1. Load Data
    Y1, Y2, C, file_names = load_all_data()
    
    d_x = 1
    d_y1 = Y1.shape[2] # 3 (x, y, z)
    d_y2 = Y2.shape[2] # 3
    d_param = C.shape[1] # 1
    time_len = Y1.shape[1] # 1000
    num_demos = Y1.shape[0]

    # 2. Load Model
    model = dual_enc_dec_cnmp.DualEncoderDecoder(d_x, d_y1, d_y2, d_param)
    model_path = os.path.join(save_path, model_name)
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print(f"Loading model state from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # 3. Select Random Indices
    # Ensure we don't crash if we have fewer than 3 trajectories
    num_to_plot = min(num_samples, num_demos)

    random.seed(42)  # For reproducibility
    indices = random.sample(range(num_demos), num_to_plot)
    
    # 4. Define Condition Points (Start and Middle)
    # We condition on t=0 (0.0) and t=500 (0.5)
    time_steps = np.linspace(0, 1, time_len)
    cond_step_indices = [0, time_len // 2] # Index 0 and 500
    
    # 5. Plot Setup
    # Grid: Rows = Trajectories, Columns = Dimensions (X, Y, Z)
    fig, axes = plt.subplots(num_to_plot, d_y1, figsize=(15, 4 * num_to_plot))
    if num_to_plot == 1: axes = np.expand_dims(axes, 0) # Handle single row case

    print(f"Evaluating indices: {indices}")

    for row_idx, traj_idx in enumerate(indices):
        
        # --- A. Prepare Input for this Trajectory ---
        # Get ground truth
        curr_y_truth = Y2[traj_idx].numpy() # (1000, 3)
        curr_file_name = file_names[traj_idx]
        
        # Create Condition Points: List of [time, value]
        # value shape must be (1, dim)
        condition_points = []
        for t_idx in cond_step_indices:
            t_val = time_steps[t_idx]
            y_val = Y1[traj_idx, t_idx:t_idx+1] # Slice to keep dims
            condition_points.append([t_val, y_val])
        
        # Expand context for the prediction function
        # C[traj_idx] is (1,) -> Expand to (1, time_len, 1)
        curr_context = C[traj_idx].view(1, 1, -1).repeat(1, time_len, 1)

        # --- B. Run Inference ---
        with torch.no_grad():
            means, stds = model_predict.predict_inverse(
                model, time_len, curr_context, condition_points, d_x, d_y1, d_y2
            )
            # means shape: (1000, 3)
            
        # --- C. Plotting ---
        dim_labels = ["X (Position)", "Y (Position)", "Z (Position)"]
        
        for col_idx in range(d_y1):
            ax = axes[row_idx, col_idx]
            
            # 1. Ground Truth
            ax.plot(time_steps, curr_y_truth[:, col_idx], 
                    color='black', linestyle='-', linewidth=2, alpha=0.5, label='Ground Truth')
            
            # 2. Prediction (Mean)
            ax.plot(time_steps, means[:, col_idx].numpy(), 
                    color='blue', linestyle='--', linewidth=2, label='Prediction')
            
            # 3. Uncertainty (Std Dev shading)
            # CNMP outputs std dev, we can shade +/- 2 sigmas
            sigma = stds[:, col_idx].numpy()
            mean_curve = means[:, col_idx].numpy()
            ax.fill_between(time_steps, mean_curve - 2*sigma, mean_curve + 2*sigma, 
                            color='blue', alpha=0.1, label='Uncertainty (2$\sigma$)')
            
            # 4. Condition Points (Red Dots)
            for cp in condition_points:
                t_c = cp[0]
                val_c = cp[1][0, col_idx]
                ax.scatter(t_c, val_c, color='red', s=60, zorder=5, edgecolors='white', linewidth=1.5)

            # Labels and Titles
            if row_idx == 0:
                ax.set_title(dim_labels[col_idx], fontsize=14, fontweight='bold')
            
            if col_idx == 0:
                ax.set_ylabel(f"Traj {traj_idx}\n({curr_file_name[:15]}...)", fontsize=10)

            ax.grid(True, alpha=0.3)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize='small', loc='best')

    plt.suptitle(f"Multi-Trajectory Evaluation (Run {run_id})\nConditioned at t=0.0 and t=0.5", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Make room for suptitle
    
    save_file = f'{save_path}/eval_multi_traj_results.png'
    plt.savefig(save_file)
    print(f"Evaluation plots saved to {save_file}")
    # plt.show()

if __name__ == "__main__":
    plot_training_progress()
    evaluate_random_trajectories(num_samples=8)