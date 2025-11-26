import torch
import numpy as np
import matplotlib.pyplot as plt
import model_mock_perfectly_paired as dual_enc_dec_cnmp
import model_predict

run_id = "run_1764155951.470604"

# Load the trained model
data_type = "perfect_paired/sin"
data_folder = f"mock_data/{data_type}"

# ============== Plot training progress ==============
# Load training progress data
training_errors = np.load(f'save/{data_type}/{run_id}/training_errors_mse.npy')
validation_errors = np.load(f'save/{data_type}/{run_id}/validation_errors_mse.npy')
losses = np.load(f'save/{data_type}/{run_id}/losses_log_prob.npy')

plt.figure(figsize=(12, 6))

# Plot errors
plt.subplot(1, 2, 1)
plt.plot(training_errors, label='Training Error', linewidth=2)
plt.plot(validation_errors, label='Validation Error', linewidth=2)
plt.xlabel('Training Step')
plt.ylabel('Error')
plt.title('Training vs Validation Error (MSE) Over Time')
plt.legend()
plt.grid(True)

# Plot losses
plt.subplot(1, 2, 2) 
plt.plot(losses, label='Training Loss', linewidth=2)
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Training Loss (Log Prob) Over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f'save/{data_type}/{run_id}/training_progress.png', dpi=150) # Save the figure
plt.show()

# ============== Plot model predictions vs actual trajectories ==============
# Load data
Y1_Paired = torch.load(f"{data_folder}/forward_modality_1_paired_data.pt", weights_only=True)
Y1_Aux = torch.load(f"{data_folder}/forward_modality_1_aux_data.pt", weights_only=True)
Y1_Inverse_Paired = torch.load(f"{data_folder}/inverse_modality_1_paired_data.pt", weights_only=True)
Y1_Inverse_Aux = torch.load(f"{data_folder}/inverse_modality_1_aux_data.pt", weights_only=True)

Y2_Paired = torch.load(f"{data_folder}/forward_modality_2_paired_data.pt", weights_only=True)
Y2_Aux = torch.load(f"{data_folder}/forward_modality_2_aux_data.pt", weights_only=True)
Y2_Inverse_Paired = torch.load(f"{data_folder}/inverse_modality_2_paired_data.pt", weights_only=True)
Y2_Inverse_Aux = torch.load(f"{data_folder}/inverse_modality_2_aux_data.pt", weights_only=True)

C1_Paired = torch.load(f"{data_folder}/context_modality_1_paired_data.pt", weights_only=True)
C1_Aux = torch.load(f"{data_folder}/context_modality_1_aux_data.pt", weights_only=True)
C2_Paired = torch.load(f"{data_folder}/context_modality_2_paired_data.pt", weights_only=True)
C2_Aux = torch.load(f"{data_folder}/context_modality_2_aux_data.pt", weights_only=True)

C1_Paired, C1_Aux, C2_Paired, C2_Aux = C1_Paired.float(), C1_Aux.float(), C2_Paired.float(), C2_Aux.float()

Y1 = torch.cat((Y1_Paired, Y1_Aux, Y2_Paired, Y2_Aux), dim=0) # Y1 is the forward trajectories of modality 1 and modality 2
Y2 = torch.cat((Y1_Inverse_Paired, Y1_Inverse_Aux, Y2_Inverse_Paired, Y2_Inverse_Aux), dim=0) # Y2 is the inverse trajectories of modality 1 and modality 2
C = torch.cat((C1_Paired, C1_Aux, C2_Paired, C2_Aux), dim=0)

# Normalize (same as training)
for dim in range(Y1.shape[2]):
    min_dim = torch.minimum(Y1[:, :, dim].min(), Y2[:, :, dim].min())
    max_dim = torch.maximum(Y1[:, :, dim].max(), Y2[:, :, dim].max())
    Y1[:, :, dim] = (Y1[:, :, dim] - min_dim) / (max_dim - min_dim)
    Y2[:, :, dim] = (Y2[:, :, dim] - min_dim) / (max_dim - min_dim)

# Normalize context to [0, 1] range
C_min = C.min()
C_max = C.max()
C = (C - C_min) / (C_max - C_min)

auxiliary_context_values = [(C1_Aux[0] - C_min)/(C_max - C_min), (C2_Aux[0] - C_min)/(C_max - C_min)]
print(auxiliary_context_values)

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

#### Test Modality 2 ####

## 1) Condition from inverse trajectory (final point), predict inverse trajectory ##
time = np.linspace(0, 1, time_len)
idx = [60, time_len - 1] # Point(s) to be used as condition
time = [time[i] for i in idx]

# Plot all trajectories
plt.figure(figsize=(50, 40))
for j in range(Y1.shape[0]):
    # Prepare input
    context = C[j:j+1]  # Shape: [1, d_param]
    x_target = X1[j:j+1]  # Shape: [1, time_len, 1]

    # Create observation (using the forward trajectory)
    i_condition_points = [[t, Y1[j, i:i+1]] for t,i in zip(time, idx)]

    # Predict inverse
    with torch.no_grad():
        ii_means, ii_stds = model_predict.predict_inverse(model, time_len, context, i_condition_points, d_x, d_y1, d_y2)
        
        # # Denormalize predicted inverse
        # ii_means = ii_means * (max_dim - min_dim) + min_dim

    plt.subplot(6, 4, j+1)
    for i in range(Y1.shape[0]):
        plt.plot(X1[0], Y1[i], color='blue' if i == j else 'orange', alpha=0.8)
        plt.plot(X1[0], Y2[i], color='blue' if i == j else 'orange', alpha=0.8)

    plt.plot(X1[0], ii_means, color='green', linewidth=2, linestyle='--', label='Predicted Inverse Trajectory')

    for i in range(len(i_condition_points)):
        plt.scatter(i_condition_points[i][0], i_condition_points[i][1], color='red', s=100, label='Condition Point' if i==0 else "")

    plt.tight_layout()
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Trajectory Value')
    plt.title(f'Model Prediction vs Actual Trajectory for Trajectory Index {j} (Modality {"1" if j < Y1_Paired.shape[0] + Y1_Aux.shape[0] else "2"}) with Context {context[0][0].numpy()} ({"Aux" if context[0].numpy() in auxiliary_context_values else "Paired"})')
    plt.legend()
plt.savefig(f'save/{data_type}/{run_id}/prediction_vs_actual.png', dpi=150)
plt.show()
