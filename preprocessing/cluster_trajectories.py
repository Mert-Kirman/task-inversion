import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Configuration
PROCESSED_FOLDER = 'data/processed_relative_high_level_actions'
ACTION_NAME = 'pick'
OBJECT_NAME = 'round_peg_4'
NUM_CLUSTERS = 2
SENSOR_TO_CLUSTER = 'pose'


def load_dataset(action_path, object_path):
    """Loads all .npy files for the specific action/object."""
    full_path = os.path.join(action_path, object_path)
    if not os.path.exists(full_path):
        print(f"Path not found: {full_path}")
        return [], []

    file_names = [f for f in os.listdir(full_path) if f.endswith('.npy')]
    data_list = []
    valid_files = []

    print(f"Loading {len(file_names)} trajectories...")

    for fname in file_names:
        try:
            file_path = os.path.join(full_path, fname)
            # Load the dictionary
            data_dict = np.load(file_path, allow_pickle=True).item()
            
            # Extract the sensor data (Shape: [1000, 7] for pose)
            # We only use the first element of the tuple (values), not timestamps
            trajectory = data_dict[SENSOR_TO_CLUSTER][0] # Shape: (1000, 7)
            
            # Use only Position (X,Y,Z) for clustering, ignore Orientation
            if SENSOR_TO_CLUSTER == 'pose':
                trajectory = trajectory[:, :3] # Shape: (1000, 3)

            data_list.append(trajectory)
            valid_files.append(fname)
            
        except Exception as e:
            print(f"Error loading {fname}: {e}")

    return np.array(data_list), valid_files # Shape: (56, 1000, 3)

def cluster_and_plot(data, filenames, n_clusters):
    """
    Clusters trajectories and plots the results.
    data shape: (N_samples, Time, Dimensions)
    """
    N, T, D = data.shape
    
    # 1. Flatten for K-Means (N, T*D)
    # We turn the whole curve into one long vector
    data_flattened = data.reshape(N, -1)
    
    # 2. Apply K-Means
    print(f"Clustering into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_flattened)
    
    # 3. Plotting
    fig, axes = plt.subplots(1, D, figsize=(6 * D, 5))
    if D == 1: axes = [axes] # Handle 1D case
    
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] # Colors for clusters
    
    time_steps = np.arange(T)
    
    print("\n--- Cluster Assignments ---")
    for cluster_id in range(n_clusters):
        # Find all files belonging to this cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        print(f"Cluster {cluster_id}: {len(cluster_indices)} trajectories")
        
        # Plot each dimension
        for dim in range(D):
            ax = axes[dim]
            
            # Plot individual trajectories in this cluster
            for idx in cluster_indices:
                ax.plot(time_steps, data[idx, :, dim], 
                        color=colors[cluster_id % len(colors)], 
                        alpha=0.3)
            
            # Plot the CENTROID (The "Average" behavior of this cluster)
            centroid = kmeans.cluster_centers_[cluster_id].reshape(T, D)
            ax.plot(time_steps, centroid[:, dim], 
                    color=colors[cluster_id % len(colors)], 
                    linewidth=3, 
                    label=f'Cluster {cluster_id} Mean')
            
            dim_name = ["X", "Y", "Z"][dim] if D == 3 else f"Dim {dim}"
            ax.set_title(f"Dimension: {dim_name}")
            ax.set_xlabel("Time")
            if dim == 0: ax.legend()

    plt.suptitle(f"Trajectory Clustering (K={n_clusters}) for {ACTION_NAME}")
    plt.tight_layout()
    plt.savefig(f'data/plots/clustering_{ACTION_NAME}_{OBJECT_NAME}.png')
    print(f"Saved clustering plot to data/plots/clustering_{ACTION_NAME}_{OBJECT_NAME}.png")
    plt.close()

    return labels

if __name__ == "__main__":
    # Construct paths
    # Assuming script is in root, adjust 'data/...' if needed
    base_path = os.path.join(PROCESSED_FOLDER, ACTION_NAME)
    
    # Load
    traj_data, file_names = load_dataset(PROCESSED_FOLDER, os.path.join(ACTION_NAME, OBJECT_NAME)) # Shape: (56, 1000, 3)
    
    if len(traj_data) > 0:
        # Cluster
        labels = cluster_and_plot(traj_data, file_names, n_clusters=NUM_CLUSTERS)
        
        # Print Example Assignment
        print("\nDetails:")
        for i in range(min(10, len(file_names))):
            print(f"  {file_names[i]} -> Cluster {labels[i]}")