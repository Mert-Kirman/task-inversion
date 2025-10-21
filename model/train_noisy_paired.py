import torch
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math
import importlib
import validate_model
import model_noisy_paired as dual_enc_dec_cnmp
import utils
importlib.reload(dual_enc_dec_cnmp)
importlib.reload(utils)
import os
from tqdm import tqdm
import sys

folders = ["1/","2/", "3/", "4/", "5/"]

for folder in folders:
    Y1_Paired = torch.tensor(np.load("data/noisy_paired/forward_noisy_paired_data.npy"), dtype=torch.float32)
    Y2_Paired = torch.tensor(np.load("data/noisy_paired/inverse_noisy_paired_data.npy"), dtype=torch.float32)

    Y1_Extra = torch.tensor(np.load("data/" + folder + "forward_extra_data.npy"), dtype=torch.float32)
    Y2_Extra = torch.tensor(np.load("data/" + folder + "inverse_extra_data.npy"), dtype=torch.float32)

    Y1 = torch.cat((Y1_Paired, Y1_Extra), dim=0)
    Y2 = torch.cat((Y2_Paired, Y2_Extra), dim=0)

    CI = torch.tensor(np.load("data/" + folder + "noisy_paired/context_pretrained_features_lower_train_dim75.npy"), dtype=torch.float32)
    C1_Paired = torch.tensor(np.load("data/noisy_paired/context_noisy_paired_data.npy"), dtype=torch.float32)[:, [0,1]]
    C1_Extra = torch.tensor(np.load("data/" + folder + "context_extra_data.npy"), dtype=torch.float32)[:, [0, 1]]

    C1 = torch.cat((C1_Paired, C1_Extra), dim=0)
    C1 = (C1 - (-0.15)) / (0.30)
    C = torch.cat((C1, CI), dim=-1)


    for dim in range(Y1.shape[2]):
        min_dim = torch.minimum(Y1[:, :, dim].min(), Y2[:, :, dim].min())
        max_dim = torch.maximum(Y1[:, :, dim].max(), Y2[:, :, dim].max())
        Y1[:, :, dim] = (Y1[:, :, dim] - min_dim) / (max_dim - min_dim)
        Y2[:, :, dim] = (Y2[:, :, dim] - min_dim) / (max_dim - min_dim)
        #print(f"Dimension {dim}: min={min_dim}, max={max_dim}")

    num_demo = Y1.shape[0]
    time_len = Y1.shape[1]

    X1 = torch.linspace(0, 1, time_len).repeat(num_demo, 1).reshape(num_demo, -1, 1)
    X2 = torch.linspace(0, 1, time_len).repeat(num_demo, 1).reshape(num_demo, -1, 1)

    valid_inverses = [True for i in range(60)] + [False for i in range(20)]

    d_x = 1
    d_param = C.shape[1]
    d_y1 = Y1.shape[2]
    d_y2 = Y2.shape[2]

    OBS_MAX = 10
    d_N = num_demo

    # last 20 traj are extra
    # poke: 60-63
    # push: 64-72
    # pick: 73-80
    validation_indices = [69, 71, 74, 76]     

    demo_data = [X1, X2, Y1, Y2, C]


    importlib.reload(dual_enc_dec_cnmp)
    importlib.reload(utils)

    def train(model, optimizer, scheduler, EPOCHS, run_id, unpaired_traj=True):
        
        os.makedirs(f'logs/{folder}run{run_id}', exist_ok=True)
        sys.stdout = open(f'logs/{folder}run{run_id}/noisy_train_log.txt', 'w')

        errors = []
        losses = []

        unpaired_traj = True 
        
        for i in tqdm(range(EPOCHS)):

            extra_pass = False
            if unpaired_traj:
                p = np.random.random_sample()
                if p < 0.20:
                    extra_pass = True

            obs, params, mask, x_tar, y_tar_f, y_tar_i, extra_pass = dual_enc_dec_cnmp.get_training_sample(extra_pass, validation_indices, 
                                                                    valid_inverses, demo_data, OBS_MAX, d_N, d_x, d_y1, d_y2, d_param, time_len)
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
                #tqdm.write(f"Run ID: {run_id}, Saved model epoch {i}, Train loss: {loss.item():6f}")
                    #torch.save(model.state_dict(), f'save/{folder}run{run_id}/perfect_paired_{i}.pth')
                    torch.save(model.state_dict(), f'save/{folder}run{run_id}/noisy_paired.pth')
                    
        # save last model
        #torch.save(model.state_dict(), f'save/{folder}run{run_id}/perfect_paired_final_{i}.pth')
    
        return errors, losses
    
    
    from torch.optim.lr_scheduler import LambdaLR

    for run_id in range(1, 3):
        os.makedirs(f'save/{folder}run{run_id}', exist_ok=True)
        errors = []
        losses = []
        errors_with_latent = []

        EPOCHS = 80_000
        learning_rate = 1e-3
        model = dual_enc_dec_cnmp.DualEncoderDecoder(d_x, d_y1, d_y2, d_param)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 if epoch < 40_000 else 5e-1)

        train(model, optimizer, scheduler, EPOCHS, run_id, unpaired_traj=True)