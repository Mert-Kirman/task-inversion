import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d

# ================= CONFIGURATION =================
INPUT_DIR = 'data/paired_trajectories_insert_place'
OUTPUT_DIR = 'data/warped_trajectories'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use Z-Only (Index 2) for alignment
# X and Y vary too much spatially (random starts), so using them for 
# temporal alignment adds noise. Z captures the shared "Lift/Drop" structure.
ALIGN_DIMS = [2] 
# =================================================

def compute_accumulated_cost_matrix(x, y):
    """
    Computes the DTW accumulated cost matrix using Euclidean distance.
    x, y: (T, D) arrays
    """
    # Pairwise distances (T x T)
    dist_mat = cdist(x, y, metric='euclidean')
    
    n_x, n_y = dist_mat.shape
    cost_mat = np.zeros((n_x, n_y))
    
    # Initialize first cell
    cost_mat[0, 0] = dist_mat[0, 0]
    
    # Initialize edges
    for i in range(1, n_x):
        cost_mat[i, 0] = dist_mat[i, 0] + cost_mat[i-1, 0]
    for j in range(1, n_y):
        cost_mat[0, j] = dist_mat[0, j] + cost_mat[0, j-1]
        
    # Fill matrix
    for i in range(1, n_x):
        for j in range(1, n_y):
            cost_mat[i, j] = dist_mat[i, j] + min(
                cost_mat[i-1, j],   # Insertion
                cost_mat[i, j-1],   # Deletion
                cost_mat[i-1, j-1]  # Match
            )
            
    return cost_mat

def backtrack_path(cost_mat):
    """Finds the optimal path from (N-1, M-1) to (0,0)."""
    i, j = cost_mat.shape[0] - 1, cost_mat.shape[1] - 1
    path = [(i, j)]
    
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            # Look back to min cost neighbor
            options = [cost_mat[i-1, j], cost_mat[i, j-1], cost_mat[i-1, j-1]]
            best_idx = np.argmin(options)
            
            if best_idx == 0: i -= 1
            elif best_idx == 1: j -= 1
            else: 
                i -= 1
                j -= 1
        path.append((i, j))
        
    return path[::-1] # Reverse to get 0 -> End

def warp_trajectory(source, path, target_len):
    """
    Warps 'source' trajectory according to the DTW path to match a target.
    path: List of (index_source, index_target) tuples.
    """
    # Extract the alignment mapping
    # We want to map the Source Time to the Target Time
    path = np.array(path)
    source_indices = path[:, 0]
    target_indices = path[:, 1]
    
    # We create a mapping: Target_Index -> Source_Index
    # Since multiple source indices might map to one target (and vice versa),
    # we interpolate to smooth it out.
    
    # Remove duplicates in target_indices to allow interpolation
    _, unique_idxs = np.unique(target_indices, return_index=True)
    target_indices_u = target_indices[unique_idxs]
    source_indices_u = source_indices[unique_idxs]
    
    # Interpolator: Give me a Target Index (0..999), I give you the corresponding Source Index
    warper = interp1d(
        target_indices_u, 
        source_indices_u, 
        kind='linear', 
        bounds_error=False, 
        fill_value=(source_indices_u[0], source_indices_u[-1]) # Clamp
    )
    
    # Generate new warped trajectory
    # The reference time is 0 to target_len
    new_indices = warper(np.arange(target_len))
    
    # Now sample the original source data at these new indices
    # We must handle floating point indices via interpolation of the source data itself
    T_src = source.shape[0]
    src_x_axis = np.arange(T_src)
    
    warped_data = np.zeros_like(source)
    
    # Warp every dimension (including those not used for alignment)
    for dim in range(source.shape[1]):
        edge_fill = (source[0, dim], source[-1, dim])
        
        dim_interpolator = interp1d(
            src_x_axis, 
            source[:, dim], 
            kind='linear', 
            bounds_error=False, 
            fill_value=edge_fill
        )
        warped_data[:, dim] = dim_interpolator(new_indices)
        
    return warped_data

def process_dataset(data_list, name="dataset"):
    print(f"\nProcessing {name} ({len(data_list)} trajectories)...")
    
    # 1. Extract Alignment Data (Z only)
    traj_arrays = [d['pose'][0][:, ALIGN_DIMS] for d in data_list]
    num_traj = len(traj_arrays)
    traj_len = traj_arrays[0].shape[0]
    
    # 2. Find Medoid (Reference Trajectory)
    # The Medoid is the trajectory with minimum DTW distance to all others.
    print("  Calculating pairwise distances to find Medoid...")
    total_costs = np.zeros(num_traj)
    
    # Optimization: Just calculate distance to mean? 
    # For accuracy, we check pairwise against a few candidates or all.
    # Let's check all vs all (56x56 is fast enough).
    
    # For simplicity/speed in this script, let's pick the one closest to the Euclidean Mean first,
    # then treat that as the reference. It's usually a safe heuristic.
    mean_traj = np.mean(np.stack(traj_arrays), axis=0)
    
    # Find real trajectory closest to the mean
    dists_to_mean = [np.linalg.norm(t - mean_traj) for t in traj_arrays]
    medoid_idx = np.argmin(dists_to_mean)
    reference_traj = traj_arrays[medoid_idx]
    
    print(f"  Reference Trajectory Index: {medoid_idx}")
    
    # 3. Warp Everyone to Reference
    warped_data_list = []
    
    print("  Warping trajectories...")
    for i, full_dict in enumerate(data_list):
        curr_pose = full_dict['pose'][0] # Full 7D pose
        curr_align_features = curr_pose[:, ALIGN_DIMS]
        
        # Calculate DTW path to Reference (DTW on Z only)
        cost_mat = compute_accumulated_cost_matrix(curr_align_features, reference_traj)
        path = backtrack_path(cost_mat)
        
        # Warp the FULL pose data (7 dims) using the path derived from X,Y,Z
        warped_pose = warp_trajectory(curr_pose, path, target_len=traj_len)
        
        # Save back structure
        new_dict = full_dict.copy()
        new_dict['pose'] = (warped_pose, full_dict['pose'][1]) # Keep old timestamps placeholder
        warped_data_list.append(new_dict)
        
    return warped_data_list, medoid_idx

def main():
    # Load
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
    
    # Verification Plot (Z-axis)
    plot_comparison(insert_data, warped_inserts, ref_ins, "Insert")
    plot_comparison(place_data, warped_places, ref_pla, "Place")

def plot_comparison(original, warped, ref_idx, title):
    plt.figure(figsize=(12, 5))
    
    # Plot Z Dimension (Index 2)
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
    plt.title(f"{title} - After DTW (Z-axis)")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'dtw_check_{title}.png'))
    print(f"Saved plot: dtw_check_{title}.png")

if __name__ == "__main__":
    main()
