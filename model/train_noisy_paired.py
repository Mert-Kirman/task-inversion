import torch
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math
import importlib
import validate_model
import model_mock_perfectly_paired as dual_enc_dec_cnmp
import utils
import os
from tqdm import tqdm
import sys
from torch.optim.lr_scheduler import LambdaLR
import time

def train(model, optimizer, scheduler, EPOCHS, unpaired_traj=True):

    os.makedirs(f'logs/{data_type}/run_{run_id}/', exist_ok=True)
    sys.stdout = open(f'logs/{data_type}/run_{run_id}/train_log.txt', 'w')

    errors = []
    losses = []

    unpaired_traj = True 
    
    for i in tqdm(range(EPOCHS)):

        extra_pass = False
        if unpaired_traj:
            p = np.random.random_sample()
            if p < 0.20:
                extra_pass = True

        obs, params, mask, x_tar, y_tar_f, y_tar_i, extra_pass = dual_enc_dec_cnmp.get_training_sample(extra_pass, valid_inverses, demo_data, OBS_MAX, d_N, d_x, d_y1, d_y2, d_param, time_len)
        optimizer.zero_grad()
        output, L_F, L_I, extra_pass = model(obs, params, mask, x_tar, extra_pass)
        
        loss = dual_enc_dec_cnmp.loss(output, y_tar_f, y_tar_i, d_y1, d_y2, d_param, L_F.squeeze(1), L_I.squeeze(1), extra_pass)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)


        optimizer.step()
        scheduler.step()

        if i > 35000 and i % 100 == 0:
            epoch_val_error = validate_model.val_only_extra(model, validation_indices, i, demo_data, 
                                                            d_x, d_y1, d_y2)
            errors.append(epoch_val_error)
            losses.append(loss.item())

            if min(errors) == errors[-1]:
                tqdm.write(f"Run ID: {run_id}, Saved model epoch {i}, Train loss: {loss.item():6f}, Validation error: {epoch_val_error:6f}")
                torch.save(model.state_dict(), f'{save_folder}/run_{run_id}/perfectly_paired.pth')

    return errors, losses


if __name__ == "__main__":
    data_type = "perfect_paired/sin"
    data_folder = f"mock_data/{data_type}"

    # Load trajectory(sensorimotor) data
    Y1_Paired = torch.load(f"{data_folder}/forward_data.pt", weights_only=True)
    Y2_Paired = torch.load(f"{data_folder}/inverse_data.pt", weights_only=True)

    Y1_Extra = torch.load(f"{data_folder}/forward_extra_data.pt", weights_only=True)
    Y2_Extra = torch.load(f"{data_folder}/inverse_extra_data.pt", weights_only=True)

    Y1 = torch.cat((Y1_Paired, Y1_Extra), dim=0)
    Y2 = torch.cat((Y2_Paired, Y2_Extra), dim=0)

    # Load context data (Task Parameters)
    C1_Paired = torch.load(f"{data_folder}/context_paired_data.pt", weights_only=True)
    C1_Extra = torch.load(f"{data_folder}/context_extra_data.pt", weights_only=True)

    C = torch.cat((C1_Paired, C1_Extra), dim=0)

    # Normalize Y1 and Y2 together
    for dim in range(Y1.shape[2]):
        min_dim = torch.minimum(Y1[:, :, dim].min(), Y2[:, :, dim].min())
        max_dim = torch.maximum(Y1[:, :, dim].max(), Y2[:, :, dim].max())
        Y1[:, :, dim] = (Y1[:, :, dim] - min_dim) / (max_dim - min_dim)
        Y2[:, :, dim] = (Y2[:, :, dim] - min_dim) / (max_dim - min_dim)

    num_demo = Y1.shape[0]
    time_len = Y1.shape[1]

    X1 = torch.linspace(0, 1, time_len).repeat(num_demo, 1).reshape(num_demo, -1, 1)
    X2 = torch.linspace(0, 1, time_len).repeat(num_demo, 1).reshape(num_demo, -1, 1)

    valid_inverses = [True for i in range(Y1_Paired.shape[0])] + [False for i in range(Y1_Extra.shape[0])]

    d_x = 1
    d_param = C.shape[1]
    d_y1 = Y1.shape[2]
    d_y2 = Y2.shape[2]

    OBS_MAX = 10
    d_N = num_demo

    # last 20 traj are extra
    validation_indices = [69, 71, 74, 76]     

    demo_data = [X1, X2, Y1, Y2, C]

    save_folder = f"save/{data_type}"
    run_id = time.time()
    os.makedirs(f'{save_folder}/run_{run_id}', exist_ok=True)

    errors = []
    losses = []
    errors_with_latent = []

    EPOCHS = 60_000
    learning_rate = 1e-3
    model = dual_enc_dec_cnmp.DualEncoderDecoder(d_x, d_y1, d_y2, d_param)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 if epoch < 40_000 else 5e-1)

    errors, losses = train(model, optimizer, scheduler, EPOCHS, unpaired_traj=True)
    np.save(f'{save_folder}/run_{run_id}/errors.npy', np.array(errors))
    np.save(f'{save_folder}/run_{run_id}/losses.npy', np.array(losses))