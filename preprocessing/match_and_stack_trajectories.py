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
PLACE_DIR = os.path.join(BASE_DIR, 'place', OBJ_NAME)

# Output Paths
OUTPUT_DIR = 'data/paired_trajectories_insert_place'
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
    # 1. Load Data
    print("--- Loading Insert Data ---")
    ins_names, ins_starts, ins_ends, ins_data = load_endpoints(INSERT_DIR)
    
    print("\n--- Loading Place Data ---")
    place_names, place_starts, place_ends, place_data = load_endpoints(PLACE_DIR)
    
    # 2. Compute Cost Matrix
    # LOGIC UPDATE: Match Insert END with Place START
    # We only use X and Y dimensions (indices :2) for matching to be robust against Z-height differences
    print("\n--- Computing Cost Matrix (Using X, Y dimensions only) ---")
    
    # insert_ends[:, :2] -> All End X,Y of inserts
    # place_starts[:, :2] -> All Start X,Y of places
    cost_matrix = cdist(ins_ends[:, :2], place_starts[:, :2], metric='euclidean')
    
    print(f"Cost Matrix Shape: {cost_matrix.shape}")
    
    # 3. Solve Assignment Problem (Hungarian Algorithm)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    print(f"\nMatched {len(row_ind)} pairs.")
    
    # 4. Construct Paired Lists
    matched_inserts = []
    matched_places = []
    matched_info = [] 
    
    for r, c in zip(row_ind, col_ind):
        # r is index in Insert list
        # c is index in Place list
        
        matched_inserts.append(ins_data[r])
        matched_places.append(place_data[c])
        
        # Calculate distance for logging
        dist = cost_matrix[r, c]
        matched_info.append({
            'insert_name': ins_names[r],
            'place_name': place_names[c],
            'match_distance': dist
        })
        
    # 5. Save Stacked Data
    save_path_ins = os.path.join(OUTPUT_DIR, 'insert_all.npy')
    save_path_place = os.path.join(OUTPUT_DIR, 'place_all.npy')
    save_path_meta = os.path.join(OUTPUT_DIR, 'pairing_info.npy')
    
    np.save(save_path_ins, np.array(matched_inserts))
    np.save(save_path_place, np.array(matched_places))
    np.save(save_path_meta, matched_info)
    
    print(f"\nSaved matched data to {OUTPUT_DIR}")
    print(f"  - {save_path_ins} ({len(matched_inserts)} trajectories)")
    print(f"  - {save_path_place} ({len(matched_places)} trajectories)")
    
    # 6. Verification Plot
    plot_verification(matched_inserts, matched_places, 10)

def plot_verification(inserts, places, num_plot=5):
    """
    Plots the Insert (Green) and Place (Red) trajectories.
    Logic Check: The END of Green should touch the START of Red.
    """
    plt.figure(figsize=(15, 6))
    
    # Plot X-Y Plane (Top-Down view of table)
    plt.subplot(1, 2, 1)
    
    for i in range(min(num_plot, len(inserts))):
        # Extract X and Y columns
        ins_pos = inserts[i]['pose'][0][:, :2]
        place_pos = places[i]['pose'][0][:, :2]
        
        # Plot Trajectories
        # Insert: Forward (Green)
        plt.plot(ins_pos[:, 0], ins_pos[:, 1], 'g-', alpha=0.6, label='Insert (Forward)' if i==0 else "")
        # Place: Inverse (Red)
        plt.plot(place_pos[:, 0], place_pos[:, 1], 'r--', alpha=0.6, label='Place (Inverse)' if i==0 else "")
        
        # Plot Connection (The "Handshake" Point)
        # Connect Insert End (-1) to Place Start (0)
        # These points should be very close if matching worked.
        plt.plot([ins_pos[-1, 0], place_pos[0, 0]], 
                 [ins_pos[-1, 1], place_pos[0, 1]], 
                 'k-', linewidth=2, marker='x', markersize=8, label='Match Point' if i==0 else "")
        
        # Mark Start of Insert (Origin/Table)
        plt.scatter(ins_pos[0, 0], ins_pos[0, 1], c='black', marker='o', s=30, zorder=5)

    plt.title(f"Top {num_plot} Matches (X-Y Plane)\nGoal: Green End touches Red Start")
    plt.xlabel("X (Relative)")
    plt.ylabel("Y (Relative)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot Z Plane (Height)
    plt.subplot(1, 2, 2)
    time_steps = np.linspace(0, 1, 1000)
    for i in range(min(num_plot, len(inserts))):
        ins_z = inserts[i]['pose'][0][:, 2]
        place_z = places[i]['pose'][0][:, 2]
        
        # Insert
        plt.plot(time_steps, ins_z, 'g-', alpha=0.5)
        # Place
        plt.plot(time_steps, place_z, 'r--', alpha=0.5)
        
        # Connection check in Z
        plt.plot([1.0, 0.0], [ins_z[-1], place_z[0]], 'k:', alpha=0.3)
        
    plt.title("Z-Height Profile\n(Note: Z might differ at connection point)")
    plt.xlabel("Time (Normalized)")
    plt.ylabel("Z Height")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_img = os.path.join(OUTPUT_DIR, 'match_verification.png')
    plt.savefig(save_img)
    print(f"Verification plot saved to {save_img}")

if __name__ == "__main__":
    match_trajectories()
