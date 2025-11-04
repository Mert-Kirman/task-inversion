import torch
import numpy as np
import matplotlib.pyplot as plt
import model_mock_perfectly_paired as dual_enc_dec_cnmp

run_id = "run_1762272182.6701496"

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

# Load trained model (update with your actual run_id)
model = dual_enc_dec_cnmp.DualEncoderDecoder(d_x, d_y1, d_y2, d_param)
model_path = f"save/{data_type}/{run_id}/perfectly_paired.pth"  # Update this!
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

# Select one forward extra trajectory to predict (index relative to extra data)
extra_idx = 0  # First extra trajectory
actual_forward_extra_idx = Y1_Paired.shape[0] + extra_idx  # Actual index in full dataset

# Prepare input
forward_traj = Y1_Extra[extra_idx:extra_idx+1]  # Shape: [1, time_len, d_y1]
context = C[actual_forward_extra_idx:actual_forward_extra_idx+1]  # Shape: [1, d_param]
context = context.unsqueeze(1).expand(-1, time_len, -1)  # Shape: [1, time_len, d_param]
x_target = X2[actual_forward_extra_idx:actual_forward_extra_idx+1]  # Shape: [1, time_len, 1]

# Create observation (using the forward trajectory)
obs_forward = torch.cat([X1[actual_forward_extra_idx:actual_forward_extra_idx+1], forward_traj], dim=-1)
obs_inverse = torch.cat([X2[actual_forward_extra_idx:actual_forward_extra_idx+1], torch.zeros_like(Y2[actual_forward_extra_idx:actual_forward_extra_idx+1])], dim=-1)

# Concatenate both observations
obs = torch.cat([obs_forward, obs_inverse], dim=-1)  # Shape: [1, time_len, d_x+d_y1+d_x+d_y2]
mask_size = time_len
mask = [
    torch.eye(mask_size).unsqueeze(0),   # Forward mask: identity matrix (each point uses itself)
    torch.zeros(1, mask_size, mask_size)  # Inverse mask: no inverse observations
]

# Predict inverse
with torch.no_grad():
    output, _, _, _ = model(obs, context, mask, x_target, extra_pass=False)
    predicted_inverse = output[:, :, d_y1:d_y1+d_y2]  # Extract inverse part

print("Predicted inverse trajectory shape:", predicted_inverse.shape)

# Get actual inverse
actual_inverse = Y2_Extra[extra_idx]

# Plot
plt.figure(figsize=(12, 6))

# Plot each dimension
for dim in range(d_y2):
    plt.subplot(1, d_y2, dim+1)
    plt.plot(actual_inverse[:, dim].numpy(), label='Actual Inverse', linewidth=2)
    plt.plot(predicted_inverse[0, :, dim].numpy(), label='Predicted Inverse', linewidth=2, linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel(f'Dimension {dim+1}')
    plt.title(f'Inverse Trajectory - Dim {dim+1}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig(f'save/{data_type}/{run_id}/prediction_vs_actual.png', dpi=150)
plt.show()

# Print error metrics
mse = torch.mean((predicted_inverse[0] - actual_inverse) ** 2).item()
print(f"Mean Squared Error: {mse:.6f}")

#plot errors.npy and losses.npy saved during training to visualize training progress
errors = np.load(f'save/{data_type}/{run_id}/errors.npy')  # Update this!
losses = np.load(f'save/{data_type}/{run_id}/losses.npy')  # Update this!

plt.figure(figsize=(12, 6))

# Plot errors
plt.subplot(1, 2, 1)
plt.plot(errors, label='Validation Error', linewidth=2)
plt.xlabel('Training Step')
plt.ylabel('Error')
plt.title('Validation Error Over Time')
plt.legend()
plt.grid(True)

# Plot losses
plt.subplot(1, 2, 2) 
plt.plot(losses, label='Training Loss', linewidth=2)
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f'save/{data_type}/{run_id}/training_progress.png', dpi=150) # Save the figure
plt.show()