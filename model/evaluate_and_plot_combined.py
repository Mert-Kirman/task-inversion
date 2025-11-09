import torch
import numpy as np
import matplotlib.pyplot as plt
import model_mock_perfectly_paired as dual_enc_dec_cnmp
import model_predict

run_id = "run_1762635634.7670999"

# Load the trained model
data_type = "perfect_paired/sin"
data_folder = f"mock_data/{data_type}"

# Load data
Y1_Paired = torch.load(f"{data_folder}/forward_data.pt", weights_only=True)
Y2_Paired = torch.load(f"{data_folder}/inverse_data.pt", weights_only=True)
Y1_Extra = torch.load(f"{data_folder}/forward_extra_data.pt", weights_only=True)
Y2_Extra = torch.load(f"{data_folder}/inverse_extra_data.pt", weights_only=True)
C1_Paired = torch.load(f"{data_folder}/context_paired_data.pt", weights_only=True)
C1_Extra = torch.load(f"{data_folder}/context_extra_data.pt", weights_only=True)

Y1 = torch.cat((Y1_Paired, Y1_Extra), dim=0)
Y2 = torch.cat((Y2_Paired, Y2_Extra), dim=0)
C = torch.cat((C1_Paired, C1_Extra), dim=0)

# Normalize (same as training)
for dim in range(Y1.shape[2]):
    min_dim = torch.minimum(Y1[:, :, dim].min(), Y2[:, :, dim].min())
    max_dim = torch.maximum(Y1[:, :, dim].max(), Y2[:, :, dim].max())
    Y1[:, :, dim] = (Y1[:, :, dim] - min_dim) / (max_dim - min_dim)
    Y2[:, :, dim] = (Y2[:, :, dim] - min_dim) / (max_dim - min_dim)

num_demo = Y1.shape[0]
time_len = Y1.shape[1]
d_x = 1
d_param = C.shape[1]
d_y1 = Y1.shape[2]
d_y2 = Y2.shape[2]

X1 = torch.linspace(0, 1, time_len).repeat(num_demo, 1).reshape(num_demo, -1, 1)
X2 = torch.linspace(0, 1, time_len).repeat(num_demo, 1).reshape(num_demo, -1, 1)

# Load trained model
model = dual_enc_dec_cnmp.DualEncoderDecoder(d_x, d_y1, d_y2, d_param)
model_path = f"save/{data_type}/{run_id}/perfectly_paired.pth"
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

# Select one forward extra trajectory to predict (index relative to extra data)
extra_idxs = [0, 20, 39]  # First extra trajectory

# Plot
plt.figure(figsize=(12, 6))
for extra_idx in extra_idxs:
    actual_forward_extra_idx = Y1_Paired.shape[0] + extra_idx  # Actual index in full dataset
    actual_forward_paired_idx = extra_idx  # Actual index in full dataset

    # Prepare input
    forward_traj_extra = Y1_Extra[extra_idx:extra_idx+1]  # Shape: [1, time_len, d_y1]
    context_extra = C[actual_forward_extra_idx:actual_forward_extra_idx+1]  # Shape: [1, d_param]
    context_extra = context_extra.unsqueeze(1).expand(-1, time_len, -1)  # Shape: [1, time_len, d_param]
    x_target = X2[actual_forward_extra_idx:actual_forward_extra_idx+1]  # Shape: [1, time_len, 1]

    forward_traj_paired = Y1_Paired[extra_idx:extra_idx+1]  # Shape: [1, time_len, d_y1]
    context_paired = C[actual_forward_paired_idx:actual_forward_paired_idx+1]  # Shape: [1, d_param]
    context_paired = context_paired.unsqueeze(1).expand(-1, time_len, -1)  # Shape: [1, time_len, d_param]
    x_target = X2[actual_forward_paired_idx:actual_forward_paired_idx+1]  # Shape: [1, time_len, 1]

    # Create observation (using the forward trajectory)
    time = np.linspace(0, 1, time_len)
    idx = np.random.permutation(time_len)
    idx = idx[:3]  # Use 3 random points as conditions
    time = [time[i] for i in idx]
    f_condition_points_extra = [[t, Y1[actual_forward_extra_idx, i:i+1]] for t,i in zip(time, idx)]
    f_condition_points_paired = [[t, Y1[actual_forward_paired_idx, i:i+1]] for t,i in zip(time, idx)]

    # Predict inverse
    with torch.no_grad():
        fi_means_extra, fi_stds_extra = model_predict.predict_inverse(model, time_len, context_extra, f_condition_points_extra, d_x, d_y1, d_y2)
        fi_means_paired, fi_stds_paired = model_predict.predict_inverse(model, time_len, context_paired, f_condition_points_paired, d_x, d_y1, d_y2)
        
        # Denormalize predicted inverse
        fi_means_extra = fi_means_extra * (max_dim - min_dim) + min_dim
        fi_means_paired = fi_means_paired * (max_dim - min_dim) + min_dim

    # Get actual inverse
    # actual_inverse = Y2_Extra[extra_idx]
    actual_inverse_extra = Y2_Extra[extra_idx]
    actual_inverse_paired = Y2_Paired[extra_idx]

    # Plot each dimension
    for dim in range(d_y2):
        # plt.subplot(1, d_y2, dim+1)
        plt.plot(forward_traj_extra[0, :, dim].numpy(), label=f'Actual Forward Extra - {extra_idx}', linewidth=2)
        plt.plot(forward_traj_paired[0, :, dim].numpy(), label=f'Actual Forward Paired - {extra_idx}', linewidth=2)
        plt.scatter([i for t,i in zip(time, idx)], [forward_traj_extra[0, i, dim].numpy() for t,i in zip(time, idx)], color='red', label='Condition Points', zorder=5)
        plt.scatter([i for t,i in zip(time, idx)], [forward_traj_paired[0, i, dim].numpy() for t,i in zip(time, idx)], color='red', label='Condition Points', zorder=5)
        plt.plot(actual_inverse_extra[:, dim].numpy(), label=f'Actual Inverse Extra - {extra_idx}', linewidth=2)
        plt.plot(actual_inverse_paired[:, dim].numpy(), label=f'Actual Inverse Paired - {extra_idx}', linewidth=2)
        plt.plot(fi_means_extra[:, dim].numpy(), label=f'Predicted Inverse Extra - {extra_idx}', linewidth=2, linestyle='--')
        plt.plot(fi_means_paired[:, dim].numpy(), label=f'Predicted Inverse Paired - {extra_idx}', linewidth=2, linestyle='--')

plt.xlabel('Time Step')
plt.ylabel(f'SM Val')
plt.title(f'Inverse Trajectory Prediction')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'save/{data_type}/{run_id}/prediction_vs_actual.png', dpi=150)
plt.show()

# # Print error metrics
# mse = torch.mean((fi_means - actual_inverse) ** 2).item()
# print(f"Mean Squared Error: {mse:.6f}")

# #plot errors.npy and losses.npy saved during training to visualize training progress
# errors = np.load(f'save/{data_type}/{run_id}/errors.npy')  # Update this!
# losses = np.load(f'save/{data_type}/{run_id}/losses.npy')  # Update this!

# plt.figure(figsize=(12, 6))

# # Plot errors
# plt.subplot(1, 2, 1)
# plt.plot(errors, label='Validation Error', linewidth=2)
# plt.xlabel('Training Step')
# plt.ylabel('Error')
# plt.title('Validation Error Over Time')
# plt.legend()
# plt.grid(True)

# # Plot losses
# plt.subplot(1, 2, 2) 
# plt.plot(losses, label='Training Loss', linewidth=2)
# plt.xlabel('Training Step')
# plt.ylabel('Loss')
# plt.title('Training Loss Over Time')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.savefig(f'save/{data_type}/{run_id}/training_progress.png', dpi=150) # Save the figure
# plt.show()

# print(fi_means[:, 0])