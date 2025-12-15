from matplotlib import pyplot as plt
import numpy as np
import os

# --- Helper Functions for Quaternion Math ---
def quaternion_inverse(q):
    """
    Computes the inverse (conjugate) of a quaternion [w, x, y, z].
    Assumes unit quaternions.
    """
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quaternion_multiply(q1, q2):
    """
    Multiplies two quaternions q1 * q2.
    Expects inputs of shape (4,) or (N, 4).
    Format: [w, x, y, z]
    """
    # Handle broadcasting if one is 1D and other is 2D
    if q1.ndim == 1: q1 = q1[np.newaxis, :]
    if q2.ndim == 1: q2 = q2[np.newaxis, :]
    
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.stack((w, x, y, z), axis=1)
# ---------------------------------------------

if __name__ == "__main__":
    robot_state_sensor_names = ['compensated_base_force', 'compensated_base_torque', 'gripper_positions', 'joint_efforts', 
                                'joint_positions', 'joint_velocities', 'measured_force', 'measured_torque', 'pose', 'velocity']
    timestamps_interpolated =  np.linspace(0, 1, 1000)
    
    processed_folder_path = 'data/processed_high_level_actions'
    available_actions = [d for d in os.listdir(processed_folder_path) if os.path.isdir(os.path.join(processed_folder_path, d))]
    
    for action in available_actions:
        action_path = os.path.join(processed_folder_path, action)
        objects = [o for o in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, o))]
        
        for obj in objects:
            object_path = os.path.join(action_path, obj)
            available_files = [f for f in os.listdir(object_path) if f.endswith('.npy')]
            
            action_object_pairs = list()
            
            print(f"Processing {action}/{obj}...")

            for file_name in available_files:
                file_path_interpolated = os.path.join(object_path, file_name)
                
                try:
                    high_level_action_dict_interpolated = np.load(file_path_interpolated, allow_pickle=True).item()
                except Exception as e:
                    print(f"Error loading {file_name}: {e}")
                    continue
                
                modality_files_interpolated = {}
                
                # --- STEP 1: TRANSFORM POSE (Position & Orientation) ---
                pose_data = high_level_action_dict_interpolated['pose'][0] # Shape (1000, 7) -> [x, y, z, qw, qx, qy, qz]
                pose_timestamps = high_level_action_dict_interpolated['pose'][1]

                num_points = pose_data.shape[0]
                search_start_idx = int(num_points * 0.60)
                
                # 1. Find grasp index (lowest Z)
                zs = pose_data[:, 2]
                local_min_idx = np.argmin(zs[search_start_idx:]) 
                grasp_index = search_start_idx + local_min_idx
                
                # --- A. Relative Position ---
                object_pos = pose_data[grasp_index, :3] # [x, y, z]
                relative_pos = pose_data[:, :3] - object_pos
                
                # --- B. Relative Orientation ---
                # Extract orientation columns [qw, qx, qy, qz] (Indices 3,4,5,6)
                quaternions = pose_data[:, 3:] 
                object_quat = quaternions[grasp_index] # The rotation at the grasp moment
                
                # Compute Q_relative = Q_current * Q_object_inverse
                object_quat_inv = quaternion_inverse(object_quat)
                relative_quat = quaternion_multiply(quaternions, object_quat_inv)
                
                # Combine back into (1000, 7)
                relative_pose = np.hstack((relative_pos, relative_quat))
                
                # Store back
                modality_files_interpolated['pose'] = (relative_pose, pose_timestamps)
                
                # Load other sensors
                for sensor in robot_state_sensor_names:
                    if sensor == 'pose': continue
                    if sensor in high_level_action_dict_interpolated:
                        val = high_level_action_dict_interpolated[sensor][0]
                        ts = high_level_action_dict_interpolated[sensor][1]
                        modality_files_interpolated[sensor] = (val, ts)

                action_object_pairs.append((file_name, modality_files_interpolated))

            # --- STEP 2: PLOTTING ---
            sensors_to_plot = ['pose']
            rows, cols = 2, 4
            data_plots_dir = f'data/plots_relative_full/{action}/{obj}'
            os.makedirs(data_plots_dir, exist_ok=True)
            
            for sensor in sensors_to_plot:
                plt.figure(figsize=(20, 15))
                
                if not action_object_pairs: continue
                    
                num_dims = action_object_pairs[0][1][sensor][0].shape[1]

                for dim in range(num_dims):
                    modality_name = f"{sensor}_{dim}"
                    
                    # Labels
                    dim_label = ""
                    if sensor == 'pose':
                        if dim == 0: dim_label = " (Rel X)"
                        elif dim == 1: dim_label = " (Rel Y)"
                        elif dim == 2: dim_label = " (Rel Z)"
                        elif dim == 3: dim_label = " (Rel Qw - Identity=1)"
                        else: dim_label = f" (Rel Q{dim-3} - Identity=0)"

                    plt.subplot(rows, cols, dim + 1)
                    
                    plot_count = 0
                    for file_name, modality_files in action_object_pairs:
                        sensor_values, _ = modality_files[sensor]
                        sensor_values_dim = sensor_values[:, dim]

                        if np.isnan(sensor_values_dim).any(): continue
                        
                        plt.plot(timestamps_interpolated, sensor_values_dim, label=f'{file_name}', alpha=0.7)
                        
                        plot_count += 1
                        if plot_count == 3: break 
                    
                    plt.title(f'{modality_name}{dim_label}')
                    plt.xlabel('Time')
                    plt.ylabel('Value')
                    plt.grid(True, alpha=0.3)
                    if dim == 0: plt.legend(fontsize='x-small')
                
                plt.suptitle(f"Fully Relative Pose (Position & Orientation)\n({action} - {obj})", fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(data_plots_dir, f'relative_{sensor}_full.png'))
                plt.close()
                print(f"Saved plot to {data_plots_dir}")