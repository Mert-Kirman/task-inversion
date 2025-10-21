import sys
import os
# Go up one directory from `simulation/` to `project_root/`
# This adds the project root to the Python path.
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

module_path = os.path.abspath(os.path.join('..', 'model'))
if module_path not in sys.path:
    sys.path.append(module_path)

import environment
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import model.model_noisy_paired as dual_enc_dec_cnmp
import model.model_predict as model_predict
from utils import *

XFORW, XINV = 0.50, 0.60
folders = ["1/", "2/", "3/", "4/", "5/"]
subfolders = ["run2/", "run1/", "run1/", "run2/", "run2/"]

errors = []
for j in range(len(folders)):

    print(f"Processing folder: {folders[j]}, subfolder: {subfolders[j]}")

    folder = folders[j]
    subfolder = subfolders[j]

    errors.append([])

    Y1_Paired = torch.tensor(np.load("data/noisy_paired/forward_noisy_paired_data.npy"), dtype=torch.float32)
    Y2_Paired = torch.tensor(np.load("data/noisy_paired/inverse_noisy_paired_data.npy"), dtype=torch.float32)

    Y1_Extra = torch.tensor(np.load("data/" + folder + "forward_extra_data.npy"), dtype=torch.float32)
    Y2_Extra = torch.tensor(np.load("data/" + folder + "inverse_extra_data.npy"), dtype=torch.float32)

    Y1 = torch.cat((Y1_Paired, Y1_Extra), dim=0)
    Y2 = torch.cat((Y2_Paired, Y2_Extra), dim=0)

    Y2_Test = np.load("data/" + folder + "inverse_test_data.npy")

    context_test_data = np.load("data/" + folder + "context_pretrained_features_lower_test_dim75.npy")
    high_level_test_context_data = np.load("data/" + folder + "high_level_test_context_data.npy")

    normalization_values = {"traj": {}, "context": {}}
    for i in range(Y1.shape[2]):
        normalization_values["traj"][i] = {
            "min": min(Y1[:, :, i].min(), Y2[:, :, i].min()),
            "max": max(Y1[:, :, i].max(), Y2[:, :, i].max())
        }
        normalization_values["context"][0] = {
            "min": -0.15,
            "max": 0.15
        }

    forward = False
    modalities = [0, 1, 2]  # 0: sphere, 1: box, 2: cylinder
    for modality in modalities:

        print(f"    Processing modality: {modality}")

        errors[-1].append([])

        indices = [i for i in range(0, 10)] if modality == 0 else ([i for i in range(10, 20)] if modality == 1 else [i for i in range(20, 30)])
        object_type = 'sphere' if modality == 0 else 'box'

        for i in indices:

            x_position = XFORW if forward else XINV
            y_position = high_level_test_context_data[i, 6]
            orientation = high_level_test_context_data[i, 3:6]
            size = high_level_test_context_data[i, :3]


            objects = [
                {'name': 'obj',
                'type': object_type,
                'pos': [x_position, y_position, 1.0],
                'quat': R.from_euler('xyz', orientation, degrees=False).as_quat().tolist(),
                'size': size,
                'rgba': [0, 0, 1, 1],  # Blue color
                'friction': [1.2, 0.5, 0.5],  # High friction for stability
                'density': 1000}
            ]

            time_len=200
            model = dual_enc_dec_cnmp.DualEncoderDecoder(d_x=1, d_y1=8, d_y2=8, d_param=75+2)
            model.load_state_dict(torch.load('../model/save/' + folder + subfolder + 'noisy_paired.pth', weights_only=True))
            model.eval()

            context_data = torch.zeros((1, 75 + 2), dtype=torch.float32)
            context_data[0, 0] = (y_position - normalization_values['context'][0]['min']) / \
                                    (normalization_values['context'][0]['max'] - normalization_values['context'][0]['min'])
            context_data[0, 1] = (y_position - normalization_values['context'][0]['min']) / \
                                    (normalization_values['context'][0]['max'] - normalization_values['context'][0]['min'])
            context_data[0, 2:] = torch.tensor(context_test_data[i, :], dtype=torch.float32)

            condition_points = []
            condition_points.append([0.0, Y1[0, 0, :]]) 

            x_position = XFORW if forward else XINV
            objects[0]['pos'] = [x_position, y_position, 1.0]

            traj = model_predict.predict_forward(
                model=model,
                time_len = time_len,
                context = context_data,
                condition_points = condition_points,
                d_x = 1,
                d_y1 = 8,
                d_y2 = 8) if forward else \
                model_predict.predict_inverse(
                model=model,
                time_len = time_len,
                context = context_data,
                condition_points = condition_points,
                d_x = 1,
                d_y1 = 8,
                d_y2 = 8)

            traj_mean, _ = traj

            unnormalized_traj = np.zeros_like(traj_mean)
            for j in range(traj_mean.shape[0]):
                for k in range(traj_mean.shape[1]):
                    unnormalized_traj[j, k] = traj_mean[j, k] *  (normalization_values['traj'][k]['max'] - 
                                                        normalization_values['traj'][k]['min']) + (
                                                        normalization_values['traj'][k]['min'])
                
            ground_truth_index = i
            ground_truth = Y2_Test[ground_truth_index, :, :7]

            #ground_truth = ground_truth / np.pi
            #ground_truth *= 180

            #unnormalized_traj = unnormalized_traj / np.pi
            #unnormalized_traj *= 180

            # mse
            mse_value = np.mean((ground_truth[:, :7] - unnormalized_traj[:, :7]) ** 2) ** 0.5
            errors[-1][-1].append(mse_value)


errors = np.array(errors)

import pandas as pd

# --- Option 1: A simple, flat table format ---

# Reshape the 3D array into a 2D array
# This will result in 5 rows (one for each model) and 30 columns (3 modalities x 10 cases)
reshaped_data = errors.reshape(5, 30)

# Create a pandas DataFrame for better labeling
# Create column headers to identify each modality and case
modalities = ['Modality A', 'Modality B', 'Modality C']
cases = [f'Case {i+1}' for i in range(10)]
columns = [f'{mod} - {case}' for mod in modalities for case in cases]

# Create row headers (index) to identify each model
index = [f'Model {i+1}' for i in range(5)]

df = pd.DataFrame(reshaped_data, index=index, columns=columns)

# --- Save to a CSV file (Recommended Method) ---
# This file can be directly opened with Excel
save_path = os.path.join('results', 'traj_errors_noisy_paired.csv')
df.to_csv(save_path)

print("Data saved to error_values.csv")
print("-" * 30)

# --- Create a copy-pastable format (Tab-separated) ---
# This is useful for direct copy-pasting
copy_paste_format = df.to_csv(sep='\t')

print("Copy the following text and paste it directly into Excel:")
print("-" * 30)
print(copy_paste_format)
     

