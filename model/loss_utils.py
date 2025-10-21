import torch
import torch.nn.functional as F
import torch.distributions as D

def log_prob_loss(output, targets_f, targets_i, d_y1, d_y2, d_param, extra_pass): # output (num_traj, num_tar, 2*d_y1 + 2*d_y2), targets (num_traj, num_tar, d_y1 + d_y2)
    
    d_sm_1 = d_y1
    d_sm_2 = d_y2

    means_f, stds_f, means_i, stds_i = output[:, :, :d_sm_1], output[:, :, d_sm_1:2*d_sm_1], output[:, :, 2*d_sm_1:2*d_sm_1+d_sm_2], output[:, :, 2*d_sm_1+d_sm_2:] # (num_traj, num_tar, d_y1 + d_y2)

    stds_f = F.softplus(stds_f)

    #dim_weights = 7 * torch.tensor([dim_weights]) / sum(dim_weights) # (1, d_y1 + d_y2)
    #dim_weights = torch.tensor([[1.0]])

    normal_f = D.Normal(means_f, stds_f)
    log_prob_f = normal_f.log_prob(targets_f[:,:,:d_sm_1]) 
    
    #weighted_log_prob_f = log_prob_f * dim_weights 

    if extra_pass:
        return -1.0 * log_prob_f.mean()
    
    stds_i = F.softplus(stds_i) # (num_traj, num_tar, d_y1)
    normal_i = D.Normal(means_i, stds_i)
    log_prob_i = normal_i.log_prob(targets_i[:,:,:d_sm_2]) 
    
    #weighted_log_prob_i = log_prob_i * dim_weights

    #total_loss = weighted_log_prob_f + weighted_log_prob_i
    total_loss = log_prob_f + log_prob_i

    return -0.5 * total_loss.mean()  # scalar

def compute_mse_of_pairs(L_F, L_I, extra_pass): #L_F/I (num_traj, 128)

    if extra_pass:
        return 0
    
    mse = 0
    i = 0

    mse += F.pairwise_distance(L_F[i], L_I[i], p=2)
    return mse

def compute_distance_trajwise(L_F, L_I, extra_pass):

    trajwise_dist = 0
    i = 0
    j = 1

    if extra_pass:
        return F.pairwise_distance(L_F[i], L_F[i], p=2)
    
    # forward
    trajwise_dist += F.pairwise_distance(L_F[j], L_F[i], p=2)
    
    trajwise_dist += F.pairwise_distance(L_I[j], L_I[i], p=2)

    return trajwise_dist / 2

def compute_norm(L_F, L_I):
    norm = 0
    for i in range(len(L_F)):
        norm += torch.norm(L_F[i]) + torch.norm(L_I[i])
    return norm

def rescale_latent_representations(L_F, L_I):
    max = torch.max(torch.max(L_F), torch.max(L_I))
    min = torch.min(torch.min(L_F), torch.min(L_I))
    # rescale between -1 and 1
    L_F = 2 * (L_F - min) / (max - min) - 1
    L_I = 2 * (L_I - min) / (max - min) - 1
    return L_F, L_I
