import os
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
BASE_DIR = 'data/processed_relative_high_level_actions'
OBJ_NAME = 'round_peg_4'

# Paths to separate directories
INSERT_DIR = os.path.join(BASE_DIR, 'insert', OBJ_NAME)
REMOVE_DIR = os.path.join(BASE_DIR, 'remove', OBJ_NAME)

# Output Paths
OUTPUT_DIR = 'data/paired_trajectories'
os.makedirs(OUTPUT_DIR, exist_ok=True)
# =================================================

def load_endpoints(directory):
    """
    Loads all .npy files in a directory and extracts:
    1. The filename
    2. The start point (t=0) x,y,z
    3. The end point (t=1) x,y,z
    4. The full pose data (to save later)
    """
    files = sorted([f for f in os.listdir(directory) if f.endswith('.npy')])
    
    filenames = []
    start_points = []
    end_points = []
    full_data = []
    
    print(f"Loading {len(files)} files from {directory}...")
    
    for f in files:
        path = os.path.join(directory, f)
        try:
            # Load dictionary
            data = np.load(path, allow_pickle=True).item()
            
            # Extract Pose (Assume shape [1000, 7])
            # [0] is values, [1] is timestamps. We take values.
            pose = data['pose'][0] 
            
            filenames.append(f)
            start_points.append(pose[0, :3])  # x,y,z at Start
            end_points.append(pose[-1, :3])   # x,y,z at End
            full_data.append(data)
            
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    return filenames, np.array(start_points), np.array(end_points), full_data

def match_trajectories():
    # Load Data
    print("--- Loading Insert Data ---")
    ins_names, ins_starts, ins_ends, ins_data = load_endpoints(INSERT_DIR)
    
    print("\n--- Loading Remove Data ---")
    rem_names, rem_starts, rem_ends, rem_data = load_endpoints(REMOVE_DIR)
    
    # Compute Cost Matrix
    # We want to match: Insert START <--> Remove END
    # This closes the geometric loop (Home -> Socket -> Home)
    print("\n--- Computing Cost Matrix ---")
    # cdist computes distance between every Insert Start and every Remove End
    # Shape: (Num_Insert, Num_Remove)
    cost_matrix = cdist(ins_starts, rem_ends, metric='euclidean')
    
    print(f"Cost Matrix Shape: {cost_matrix.shape}")
    
    # Solve Assignment Problem (Hungarian Algorithm)
    # linear_sum_assignment finds the indices that minimize the total cost.
    # It automatically handles non-square matrices by ignoring extra rows/cols.
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    print(f"\nMatched {len(row_ind)} pairs.")
    
    # Construct Paired Lists
    matched_inserts = []
    matched_removes = []
    matched_info = [] # To store filenames for verification
    
    for r, c in zip(row_ind, col_ind):
        # r is index in Insert list
        # c is index in Remove list
        
        matched_inserts.append(ins_data[r])
        matched_removes.append(rem_data[c])
        
        # Calculate distance for logging
        dist = cost_matrix[r, c]
        matched_info.append({
            'insert_name': ins_names[r],
            'remove_name': rem_names[c],
            'match_distance': dist
        })
        
    # Save Stacked Data
    # Stack dictionaries is hard, usually we save lists of dictionaries 
    # OR we stack the sensor arrays if we know the structure.
    # For simplicity/compatibility with your loader, let's save the list of dicts.
    
    save_path_ins = os.path.join(OUTPUT_DIR, 'insert_all.npy')
    save_path_rem = os.path.join(OUTPUT_DIR, 'remove_all.npy')
    save_path_meta = os.path.join(OUTPUT_DIR, 'pairing_info.npy')
    
    np.save(save_path_ins, np.array(matched_inserts))
    np.save(save_path_rem, np.array(matched_removes))
    np.save(save_path_meta, matched_info)
    
    print(f"\nSaved matched data to {OUTPUT_DIR}")
    print(f"  - {save_path_ins} ({len(matched_inserts)} trajectories)")
    print(f"  - {save_path_rem} ({len(matched_removes)} trajectories)")
    
    # Verification Plot
    # Plot the first 5 matches to visually verify "Loop Closure"
    plot_verification(matched_inserts, matched_removes, 1)

def plot_verification(inserts, removes, num_plot=5):
    """
    Plots the Insert (Green) and Remove (Red) trajectories.
    If matched correctly, Green Start should be near Red End.
    """
    plt.figure(figsize=(15, 5))
    
    # Plot X-Y Plane (Top-Down view of table)
    plt.subplot(1, 2, 1)
    
    for i in range(num_plot):
        # Extract X and Y columns
        ins_pos = inserts[i]['pose'][0][:, :2]
        rem_pos = removes[i]['pose'][0][:, :2]
        
        # Plot Trajectories
        plt.plot(ins_pos[:, 0], ins_pos[:, 1], 'g-', alpha=0.5, label='Insert (Forward)' if i==0 else "")
        plt.plot(rem_pos[:, 0], rem_pos[:, 1], 'r--', alpha=0.5, label='Remove (Inverse)' if i==0 else "")
        
        # Plot Connection (The Gap)
        # Connect Insert Start (0) to Remove End (-1)
        plt.plot([ins_pos[0, 0], rem_pos[-1, 0]], 
                 [ins_pos[0, 1], rem_pos[-1, 1]], 
                 'k:', linewidth=1, label='Match Gap' if i==0 else "")
        
        # Mark Start of Insert (Home)
        plt.scatter(ins_pos[0, 0], ins_pos[0, 1], c='k', marker='o', s=20)

    plt.title(f"Top {num_plot} Matches (X-Y Plane)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    
    # Plot Z Plane (Height)
    plt.subplot(1, 2, 2)
    time = np.linspace(0, 1, 1000)
    for i in range(num_plot):
        ins_z = inserts[i]['pose'][0][:, 2]
        rem_z = removes[i]['pose'][0][:, 2]
        
        plt.plot(time, ins_z, 'g-', alpha=0.5)
        plt.plot(time, rem_z, 'r--', alpha=0.5)
        
    plt.title("Z-Height Profile")
    plt.xlabel("Time")
    plt.ylabel("Z Height")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'match_verification.png'))
    print("Verification plot saved.")

if __name__ == "__main__":
    match_trajectories()
