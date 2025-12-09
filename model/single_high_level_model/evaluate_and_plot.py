import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
import matplotlib.pyplot as plt
import model.single_high_level_model.dual_enc_dec_model as dual_enc_dec_cnmp
import model.model_predict as model_predict

# Configuration
run_id = "run_1764627050.657804"
save_path = f"model/single_high_level_model/save/{run_id}"
data_path = "data/processed_high_level_actions"
file_name_robot = "synchronized_high_level_action_31_robot_state.npy"
file_name_time = "synchronized_high_level_action_31_timestamps.npy"

def load_and_process_data():
    """
    Loads data exactly as train.py does to ensure consistency.
    """
    print("Loading data...")
    high_level_action_dict = np.load(os.path.join(data_path, file_name_robot), allow_pickle=True).item()
    timestamps_dict = np.load(os.path.join(data_path, file_name_time), allow_pickle=True)
    
    robot_state_sensor_names = ['compensated_base_force', 'compensated_base_torque', 'gripper_positions', 
                                'joint_efforts', 'joint_positions', 'joint_velocities', 
                                'measured_force', 'measured_torque', 'pose', 'velocity']
    
    modality_files = {}
    for sensor in robot_state_sensor_names:
        sensor_values = high_level_action_dict[sensor]
        timestamps = timestamps_dict
        modality_files[sensor] = (sensor_values, timestamps)

    selected_sensors = ['joint_positions', 'pose', 'gripper_positions', 'compensated_base_force']
    
    # Concatenate selected sensors
    Y1 = torch.tensor(np.concatenate([modality_files[sensor][0] for sensor in selected_sensors], axis=1), dtype=torch.float32)
    
    # Handle NaNs (Sanitization)
    if torch.isnan(Y1).any():
        print("WARNING: Input data contained NaNs. Replacing with 0.0.")
        Y1 = torch.nan_to_num(Y1, nan=0.0)

    Y1 = Y1.unsqueeze(0)  # Add batch dimension: (1, 1000, 19)
    Y2 = Y1.detach().clone() # In this single trajectory setup, Inverse is same as Forward
    
    # Dummy context
    C = torch.zeros((1, 1)) 

    # Normalize (Must match train.py logic exactly)
    min_vals = []
    max_vals = []
    
    for dim in range(Y1.shape[2]):
        min_dim = torch.minimum(Y1[:, :, dim].min(), Y2[:, :, dim].min())
        max_dim = torch.maximum(Y1[:, :, dim].max(), Y2[:, :, dim].max())
        
        # Store for potential denormalization later
        min_vals.append(min_dim)
        max_vals.append(max_dim)

        denominator = max_dim - min_dim
        
        if denominator == 0:
            Y1[:, :, dim] = 0.0 
            Y2[:, :, dim] = 0.0
        else:
            Y1[:, :, dim] = (Y1[:, :, dim] - min_dim) / denominator
            Y2[:, :, dim] = (Y2[:, :, dim] - min_dim) / denominator

    return Y1, Y2, C, min_vals, max_vals

def plot_training_progress():
    try:
        training_errors = np.load(f'{save_path}/training_errors_mse.npy')
        losses = np.load(f'{save_path}/losses_log_prob.npy')

        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(training_errors, label='Training MSE', color='blue')
        plt.title('Training Error (MSE)')
        plt.xlabel('Validation Step')
        plt.ylabel('MSE')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(losses, label='Training Loss', color='orange')
        plt.title('Training Loss (Log Prob)')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{save_path}/training_progress.png')
        print("Training progress saved.")
        plt.show()
    except FileNotFoundError:
        print("Training logs not found, skipping progress plot.")

def evaluate_model():
    # Load Data
    Y1, Y2, C, min_vals, max_vals = load_and_process_data()
    
    d_x = 1
    d_param = C.shape[1]
    d_y1 = Y1.shape[2] # 19
    d_y2 = Y2.shape[2] # 19
    time_len = Y1.shape[1] # 1000

    X1 = torch.linspace(0, 1, time_len).reshape(1, time_len, 1)

    # Load Model
    model = dual_enc_dec_cnmp.DualEncoderDecoder(d_x, d_y1, d_y2, d_param)
    model_path = f"{save_path}/perfectly_paired.pth"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Setup Inference Conditions
    # Condition on start (0) and middle (500)
    time_steps = np.linspace(0, 1, time_len)
    cond_indices = [0, time_len // 2] 
    
    # Prepare condition points from Forward Trajectory (Y1)
    # List of [time_val, sensor_values_at_time]
    condition_points = [[time_steps[i], Y1[0, i:i+1]] for i in cond_indices]
    
    # Context needs to be expanded for the predict function expectations
    # Shape: (1, time_len, d_param)
    context_expanded = C.unsqueeze(1).repeat(1, time_len, 1)

    print("Running inference...")
    with torch.no_grad():
        # We use predict_inverse to test the reconstruction/inversion capability
        # output means: (time_len, d_y2)
        means, stds = model_predict.predict_inverse(
            model, time_len, context_expanded, condition_points, d_x, d_y1, d_y2
        )

    # Plotting
    # We have 19 dimensions. Let's make a grid.
    sensor_names = (
        [f'Joint {i}' for i in range(7)] + 
        [f'Pose {i}' for i in range(7)] + 
        [f'Gripper {i}' for i in range(2)] + 
        [f'Force {i}' for i in range(3)]
    )

    rows = 5
    cols = 4
    plt.figure(figsize=(20, 15))

    for dim in range(d_y1):
        plt.subplot(rows, cols, dim + 1)
        
        # Plot Ground Truth
        plt.plot(time_steps, Y2[0, :, dim].numpy(), label='Ground Truth', color='blue', alpha=0.6, linewidth=2)
        
        # Plot Prediction
        plt.plot(time_steps, means[:, dim].numpy(), label='Prediction', color='green', linestyle='--', linewidth=2)
        
        # Plot Condition Points
        for cp in condition_points:
            t_cond = cp[0]
            val_cond = cp[1][0, dim]
            plt.scatter(t_cond, val_cond, color='red', s=50, zorder=5)

        plt.title(sensor_names[dim])
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.suptitle(f"Single Trajectory Reconstruction (Run {run_id})\nRed dots = Condition Points", fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_path}/prediction_results.png')
    print("Prediction results saved.")
    plt.show()

if __name__ == "__main__":
    plot_training_progress()
    evaluate_model()