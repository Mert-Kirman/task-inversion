import torch
import torch.nn as nn
import model.loss_utils as loss_utils
import torchvision.transforms as T
from torchvision.transforms import functional as F

class CustomAugmentation:
    def __init__(self):
        # Define a sequence of transformations.
        # The key is to keep the values small so you don't change the object's identity.
        self.transform = T.Compose([
            # T.ToTensor() is usually the first step to convert a PIL Image or NumPy array to a tensor.
            # Assuming your input is already a tensor, we can use RandomApply.
            
            # Apply random affine transformation (translation, rotation) with a certain probability
            T.RandomApply([
                T.RandomAffine(
                    degrees=2,          # Tiny rotation: -2 to +2 degrees
                    translate=(0.05, 0.05) # Tiny shift: up to 5% of image width/height
                )
            ], p=0.7), # Apply this 70% of the time

            # Add Gaussian noise
            # We need a custom lambda function for this as it's not a standard transform on its own.
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01) # Add noise with a small std dev
        ])

    def __call__(self, image_tensor):
        return self.transform(image_tensor)

class DualEncoderDecoder(nn.Module):
    def __init__(self, d_x, d_y1, d_y2, d_param, dropout_p=[0.5, 0.1]):
        super(DualEncoderDecoder, self).__init__()

        self.d_x = d_x
        self.d_y1 = d_y1
        self.d_y2 = d_y2
        self.param_dim = d_param
        self.learned_param_dim = d_param
        self.augmentation = CustomAugmentation()

        self.param_cnn1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.param_cnn2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.param_cnn3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.param_cnn_head1 = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Dropout(p=dropout_p[0]),
            nn.Linear(32*4*5, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(p=dropout_p[1]),
            #nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(p=dropout_p[1]),
            nn.Linear(256, self.learned_param_dim)
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_param-1, 64), nn.LayerNorm(64), nn.ReLU(), 
            nn.Linear(64, self.learned_param_dim) #, nn.LayerNorm(128), nn.ReLU(),
            #nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(),
            #nn.Linear(64, 32)
        )

        # --- Encoders with BatchNorm and Dropout ---
        # Linear -> BatchNorm -> Activation -> Dropout
        self.encoder1 = nn.Sequential(
            nn.Linear(d_x + d_y1, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 256)
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(d_x + d_y2, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 256)
        )

        # --- Decoders with BatchNorm and Dropout ---
        self.decoder1 = nn.Sequential(
            nn.Linear(d_x + 256 + (self.learned_param_dim), 256), nn.LayerNorm(256), nn.ReLU(), 
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, (d_y1)*2)
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(d_x + 256 + (self.learned_param_dim), 256), nn.LayerNorm(256), nn.ReLU(), 
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, (d_y2)*2)
        )

    def without_skip_connection_cnn(self, in1):
        # param_depth: (num_traj*num_tar, 1, 50, 50)
        in2 = self.param_cnn1(in1)
        in3 = self.param_cnn2(in2)  # (num_traj*num_tar, 64, 12, 10)
        in4 = self.param_cnn3(in3)  # (num_traj*num_tar, 128, 8, 5)
        out = self.param_cnn_head1(in4)
        out = out.unsqueeze(1)  # (num_traj*num_tar, 128)
        return out  # (num_traj*num_tar, 4)

    def forward(self, obs, params, mask, x_tar, extra_pass, p=0):
        # obs: (num_traj, max_obs_num, 2*d_x + d_y1 + d_y2) 
        # mask: (num_traj, max_obs_num, 1)
        # x_tar: (num_traj, num_tar, d_x)

        mask_forward, mask_inverse = mask[0], mask[1] # (num_traj, max_obs_num, max_obs_num)

        obs_f = obs[:, :, :self.d_x+self.d_y1]  # (num_traj, max_obs_num, d_x + d_y1)
        obs_i = obs[:, :, self.d_x+self.d_y1:2*self.d_x+self.d_y1+self.d_y2]  # (num_traj, max_obs_num, d_x + d_y2)

        param_loc = params[:, :, [0]] # (num_traj, num_tar, 4)
        param_depth = params[:, :, 1:] # (num_traj, num_tar, 50*50)

        #param_depth = param_depth.reshape(-1, 1, 40, 32)  # (num_traj*num_tar, 1, 50, 50)
        #param_depth = self.augmentation(param_depth)

        r1 = self.encoder1(obs_f)  # (num_traj, max_obs_num, 128)
        masked_r1 = torch.bmm(mask_forward, r1) # (num_traj, max_obs_num, 128)
        sum_masked_r1 = torch.sum(masked_r1, dim=1) # (num_traj, 128)
        L_F = sum_masked_r1 / (torch.sum(mask_forward, dim=[1,2]).reshape(-1,1) + 1e-10)
        L_F = L_F.unsqueeze(1).expand(-1, x_tar.shape[1], -1) # (num_traj, num_tar, 128)

        if not extra_pass:
            r2 = self.encoder2(obs_i)  # (num_traj, max_obs_num, 128)
            masked_r2 = torch.bmm(mask_inverse, r2)
            sum_masked_r2 = torch.sum(masked_r2, dim=1) # (num_traj, 128)
            L_I = sum_masked_r2 / (torch.sum(mask_inverse, dim=[1,2]).reshape(-1,1) + 1e-10)
            L_I = L_I.unsqueeze(1).expand(-1, x_tar.shape[1], -1) # (num_traj, num_tar, 128)

        latent = torch.zeros(0)
        if p == 0:
            p1 = torch.rand(1)
            p2 = torch.rand(1)
            p1 = p1 / (p1 + p2)
            if not extra_pass:
                latent = L_F * p1 + L_I * (1-p1)  # (num_traj, num_tar, 128)
            else:
                latent = L_F
        elif p == 1:
            latent = L_F # (1, num_tar, 128) , used for validation pass
        elif p == 2:
            latent = L_I


        #param_depth = self.without_skip_connection_cnn(param_depth)  # (num_traj, num_tar, 4)
        param_depth = self.mlp(param_depth)  # (num_traj, num_tar, 4)
        param_depth = param_depth.expand(-1, x_tar.shape[1], -1)  # (num_traj, num_tar, 4)
        param_loc = param_loc.expand(-1, x_tar.shape[1], -1)  # (num_traj, num_tar, 1)
        param = torch.cat((param_depth, param_loc), dim=-1)
        param = params.expand(-1, x_tar.shape[1], -1)  # (num_traj, num_tar, d_param)
        latent_with_par = torch.cat((latent, param), dim=-1)  # (num_traj, num_tar, 128 + 1)
        concat = torch.cat((latent_with_par, x_tar), dim=-1)  # (num_traj, num_tar, 128 + d_x) 
        #concat = torch.cat((latent, x_tar), dim=-1)  # (num_traj, num_tar, 128 + d_x)
        
        output1 = self.decoder1(concat)  # (num_traj, num_tar, 2*d_y1)

        if extra_pass:
            return torch.cat((output1, output1), dim=-1), L_F, L_F, extra_pass

        output2 = self.decoder2(concat)  # (num_traj, num_tar, 2*d_y2)
        # (num_traj, num_tar, 2*d_y1 + 2*d_y2)
        return torch.cat((output1, output2), dim=-1), L_F, L_I, extra_pass
    
def get_training_sample(extra_pass, validation_indices, valid_inverses, demo_data, 
                        OBS_MAX, d_N, d_x, d_y1, d_y2, d_param, time_len):

    X1, X2, Y1, Y2, C = demo_data
    
    num_traj = 24
    
    traj_multinom = torch.ones(d_N) # multinomial distribution for trajectories

    for i in range(d_N):
        if i in validation_indices:
            traj_multinom[i] = 0

    if not extra_pass:
        for i in range(len(traj_multinom)):
            if not valid_inverses[i]:
                traj_multinom[i] = 0
    
    traj_indices = torch.multinomial(traj_multinom, num_traj, replacement=False) # random indices of trajectories

    obs_num_list = torch.randint(0, OBS_MAX, (2*num_traj,)) + 1  # random number of obs. points
    max_obs_num = OBS_MAX
    observations = torch.zeros((num_traj, max_obs_num, 2*d_x + d_y1 + d_y2))
    mask_forward = torch.zeros((num_traj, max_obs_num, max_obs_num))
    mask_inverse = torch.zeros((num_traj, max_obs_num, max_obs_num))

    params = torch.zeros((num_traj, 1, d_param))
    target_X = torch.zeros((num_traj, 1, d_x))
    target_Y1 = torch.zeros((num_traj, 1, d_y1))
    target_Y2 = torch.zeros((num_traj, 1, d_y2))

    T = torch.ones(time_len)
    for i in range(num_traj):
        traj_index = int(traj_indices[i])
        obs_num_f = int(obs_num_list[i])
        obs_num_i = int(obs_num_list[num_traj + i])

        params[i] = C[traj_index]
                      
        obs_indices_f = torch.multinomial(T, obs_num_f, replacement=False)
        obs_indices_i = torch.multinomial(T, obs_num_i, replacement=False)

        for j in range(obs_num_f):
            observations[i][j][:d_x] = X1[0][obs_indices_f[j]]
            observations[i][j][d_x:d_x+d_y1] = Y1[traj_index][obs_indices_f[j]]
            mask_forward[i][j][j] = 1

        for j in range(obs_num_i):
            if valid_inverses[traj_index]:
                observations[i][j][d_x + d_y1:2*d_x + d_y1] = X2[0][obs_indices_i[j]]
                observations[i][j][2*d_x + d_y1:] = Y2[traj_index][obs_indices_i[j]]
                mask_inverse[i][j][j] = 1
        
        target_index = torch.multinomial(T, 1)
        target_X[i] = X1[0][target_index]
        target_Y1[i] = Y1[traj_index][target_index]
        target_Y2[i] = Y2[traj_index][target_index]
        
    return observations, params, [mask_forward, mask_inverse], target_X, target_Y1, target_Y2, extra_pass
    
def loss(output, target_f, target_i, d_y1, d_y2, d_param, L_F, L_I, extra_pass):

    #L_F, L_I = loss_utils.rescale_latent_representations(L_F, L_I)
    #mse_of_pairs = loss_utils.compute_mse_of_pairs(L_F, L_I, extra_pass) # scalar
    #distance_trajwise = loss_utils.compute_distance_trajwise(L_F, L_I, extra_pass) #scalar
    
    log_prob = loss_utils.log_prob_loss(output, target_f, target_i, d_y1, d_y2, d_param, extra_pass) # scalar
 
    lambda1, lambda2, lambda3 = 1, 0, 0

    return lambda1 * log_prob #+ lambda2 * mse_of_pairs + lambda3 * torch.clamp(1-distance_trajwise, min=0)


