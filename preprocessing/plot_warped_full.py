import numpy as np
import matplotlib.pyplot as plt
import os

# ================= CONFIGURATION =================
# Path to the warped data files
WARPED_DATA_DIR = 'data/warped_trajectories'
OUTPUT_PLOT_DIR = 'data/plots_warped_full'
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
# =================================================

def plot_action_data(data_list, action_name):
    """
    Plots all dimensions of the pose sensor for a given action list.
    """
    if len(data_list) == 0:
        print(f"No data found for {action_name}")
        return

    # Extract the first trajectory to get dimensions
    # Structure: dict['pose'][0] is the value array (T, D)
    first_traj = data_list[0]['pose'][0]
    num_dims = first_traj.shape[1]
    num_samples = first_traj.shape[0]
    
    # Common timestamps for plotting (normalized 0 to 1)
    timestamps = np.linspace(0, 1, num_samples)

    # Setup Figure
    rows, cols = 2, 4
    plt.figure(figsize=(20, 15))
    
    sensor = 'pose'
    
    for dim in range(num_dims):
        # Create Subplot
        plt.subplot(rows, cols, dim + 1)
        
        # Labeling
        dim_label = ""
        if dim == 0: dim_label = " (Rel X)"
        elif dim == 1: dim_label = " (Rel Y)"
        elif dim == 2: dim_label = " (Rel Z)"
        elif dim == 3: dim_label = " (Rel Qw)"
        else: dim_label = f" (Rel Q{dim-3})"
        
        # Plot every trajectory
        for i, traj_dict in enumerate(data_list):
            # dict['pose'][0] is values, [1] is timestamps (unused here)
            pose_values = traj_dict['pose'][0]
            
            # Safety check for shape
            if pose_values.shape[1] <= dim: continue
                
            val_dim = pose_values[:, dim]
            
            # Plot
            plt.plot(timestamps, val_dim, alpha=0.5, linewidth=1)
            
            # Limit legend/clutter if too many
            if i > 55: break

        plt.title(f'{sensor}_{dim}{dim_label}')
        plt.xlabel('Time (Normalized)')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(f"Warped Relative Pose - {action_name.upper()}\n(Aligned by Z-Derivative)", fontsize=16)
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(OUTPUT_PLOT_DIR, f'warped_{action_name}_full.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")

def main():
    # 1. Load Data
    insert_path = os.path.join(WARPED_DATA_DIR, 'insert_all.npy')
    place_path = os.path.join(WARPED_DATA_DIR, 'place_all.npy')
    
    if not os.path.exists(insert_path) or not os.path.exists(place_path):
        print(f"Error: Could not find .npy files in {WARPED_DATA_DIR}")
        return

    print(f"Loading warped data from {WARPED_DATA_DIR}...")
    insert_data = np.load(insert_path, allow_pickle=True)
    place_data = np.load(place_path, allow_pickle=True)
    
    print(f"Found {len(insert_data)} Insert and {len(place_data)} Place trajectories.")
    
    # 2. Plot Insert
    plot_action_data(insert_data, "insert")
    
    # 3. Plot Place
    plot_action_data(place_data, "place")

if __name__ == "__main__":
    main()
