import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from fastdtw import fastdtw

# ================= CONFIGURATION =================
INPUT_DIR = 'data/paired_trajectories_insert_place'
OUTPUT_DIR = 'data/warped_trajectories'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use Z-Only (Index 2) for alignment
ALIGN_DIMS = [2] 
# =================================================

def get_smooth_warping_indices(path, target_len, sigma=2.0):
    """
    Converts a DTW path into a SMOOTH mapping function.
    
    Args:
        sigma: Strength of smoothing. Higher = smoother time flow, fewer spikes.
               Sigma=2.0 is usually a sweet spot.
    """
    path = np.array(path)
    src_idxs = path[:, 0]
    ref_idxs = path[:, 1]
    
    # 1. Resolve Path Conflicts (Group by Reference Index)
    unique_refs, unique_indices = np.unique(ref_idxs, return_index=True)
    
    # Calculate the mean source index for each unique reference index
    # (A faster vectorizable way than loop)
    # But for clarity/robustness with varying counts, a simple loop is fine or using weighted bincount
    
    # Faster equivalent of the previous loop:
    mapped_src_idxs = []
    for ref_t in unique_refs:
        # Get all source indices matching this ref index
        matching = src_idxs[ref_idxs == ref_t]
        mapped_src_idxs.append(np.mean(matching))
    
    mapped_src_idxs = np.array(mapped_src_idxs)
    
    # 2. Create Base Interpolator
    # Maps Ref Time -> Src Time (Rough)
    warper_rough = interp1d(unique_refs, mapped_src_idxs, kind='linear', fill_value="extrapolate")
    raw_indices = warper_rough(np.arange(target_len))
    
    # 3. APPLY SMOOTHING (The Fix for Spikes)
    # This ensures time doesn't jump instantly.
    smooth_indices = gaussian_filter1d(raw_indices, sigma=sigma)
    
    # 4. Enforce Monotonicity (Time cannot go backward)
    smooth_indices = np.maximum.accumulate(smooth_indices)
    
    # 5. Safety Clamp
    smooth_indices = np.clip(smooth_indices, 0, target_len - 1)
    
    return smooth_indices

def warp_trajectory_robust(source, new_indices):
    """
    Samples the source trajectory at the given 'new_indices'.
    """
    T_src = source.shape[0]
    src_x_axis = np.arange(T_src)
    warped_data = np.zeros_like(source)
    
    for dim in range(source.shape[1]):
        dim_interpolator = interp1d(
            src_x_axis, 
            source[:, dim], 
            kind='linear', 
            bounds_error=False, 
            fill_value=(source[0, dim], source[-1, dim])
        )
        warped_data[:, dim] = dim_interpolator(new_indices)
        
    return warped_data

def process_dataset(data_list, name="dataset"):
    print(f"\nProcessing {name} ({len(data_list)} trajectories)...")
    
    # 1. Extract Alignment Data (Z Only)
    traj_arrays = [d['pose'][0][:, ALIGN_DIMS] for d in data_list]
    num_traj = len(traj_arrays)
    traj_len = traj_arrays[0].shape[0]
    
    # --- DERIVATIVE PRE-CALCULATION ---
    # We warp based on Shape (Velocity), not Absolute Position
    traj_derivatives = []
    for t in traj_arrays:
        # Calculate gradients (velocity)
        # axis=0 is time
        grad = np.gradient(t, axis=0)
        # Normalize to prevent magnitude bias
        # (Optional, but often helps. Let's stick to raw derivative first as Z-scale is consistent)
        traj_derivatives.append(grad)
    
    # 2. Find Medoid using DERIVATIVES
    print("  Calculating pairwise DTW distances (on Derivatives) to find Medoid...")
    
    # Use Euclidean Mean of the RAW data to find a representative candidate
    # (Using mean of derivatives is messy, better to find the 'average' position profile
    # and align to that).
    eucl_mean_traj = np.mean(np.stack(traj_arrays), axis=0)
    dists_to_mean = [np.linalg.norm(t - eucl_mean_traj) for t in traj_arrays]
    medoid_idx = np.argmin(dists_to_mean)
    
    # Reference is the DERIVATIVE of the medoid
    reference_deriv = traj_derivatives[medoid_idx]
    
    print(f"  Reference Trajectory Index: {medoid_idx}")
    
    # 3. Warp Everyone
    warped_data_list = []
    
    print("  Warping trajectories...")
    for i, full_dict in enumerate(data_list):
        curr_pose = full_dict['pose'][0]
        
        # Align using Derivative of Z
        curr_deriv = traj_derivatives[i]
        
        # FastDTW on Derivatives
        distance, path = fastdtw(curr_deriv, reference_deriv, dist=euclidean)
        
        # Get SMOOTH indices
        new_indices = get_smooth_warping_indices(path, target_len=traj_len, sigma=5.0)
        
        # Warp the original Pose (Position) using the map derived from Velocity
        warped_pose = warp_trajectory_robust(curr_pose, new_indices)
        
        new_dict = full_dict.copy()
        new_dict['pose'] = (warped_pose, full_dict['pose'][1])
        warped_data_list.append(new_dict)
        
    return warped_data_list, medoid_idx

def main():
    print(f"Loading from {INPUT_DIR}...")
    insert_data = np.load(os.path.join(INPUT_DIR, 'insert_all.npy'), allow_pickle=True)
    place_data = np.load(os.path.join(INPUT_DIR, 'place_all.npy'), allow_pickle=True)
    
    # Process
    warped_inserts, ref_ins = process_dataset(insert_data, "Insert")
    warped_places, ref_pla = process_dataset(place_data, "Place")
    
    # Save
    np.save(os.path.join(OUTPUT_DIR, 'insert_all.npy'), np.array(warped_inserts))
    np.save(os.path.join(OUTPUT_DIR, 'place_all.npy'), np.array(warped_places))
    
    print(f"\nSaved warped data to {OUTPUT_DIR}")
    
    plot_comparison(insert_data, warped_inserts, ref_ins, "Insert")
    plot_comparison(place_data, warped_places, ref_pla, "Place")

def plot_comparison(original, warped, ref_idx, title):
    plt.figure(figsize=(12, 5))
    dim = 2 
    
    plt.subplot(1, 2, 1)
    for d in original:
        plt.plot(d['pose'][0][:, dim], color='gray', alpha=0.3)
    plt.plot(original[ref_idx]['pose'][0][:, dim], color='red', linewidth=2, label='Reference')
    plt.title(f"{title} - Before DTW (Z-axis)")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for d in warped:
        plt.plot(d['pose'][0][:, dim], color='blue', alpha=0.3)
    plt.plot(warped[ref_idx]['pose'][0][:, dim], color='red', linewidth=2, label='Reference')
    plt.title(f"{title} - After DTW (Z-axis)\n(Aligned by Derivative + Smoothed)")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'dtw_check_{title}.png'))
    print(f"Saved plot: dtw_check_{title}.png")

if __name__ == "__main__":
    main()