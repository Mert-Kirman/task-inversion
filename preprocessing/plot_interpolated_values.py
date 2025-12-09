from matplotlib import pyplot as plt
import numpy as np
import os

def plot_interpolated_values(timestamps, sensor_values, target_timestamps, interpolated_values, modality_name, data_plots_dir):
    """
    Plot original and interpolated sensor values for comparison.
    
    Args:
        timestamps: original timestamps of the sensor data
        sensor_values: original sensor values
        target_timestamps: timestamps used for interpolation
        interpolated_values: interpolated sensor values
        modality_name: name of the sensor modality for labeling
    """
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, sensor_values, 'o', label='Original Data', markersize=5)
    plt.plot(target_timestamps, interpolated_values, '-', label='Interpolated Data')
    plt.title(f'Interpolation of {modality_name} Sensor Data')
    plt.xlabel('Time')
    plt.ylabel('Sensor Values')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(data_plots_dir, f'interpolated_{modality_name}.png'))
    # plt.show()
    plt.close()

if __name__ == "__main__":
    robot_state_sensor_names = ['compensated_base_force', 'compensated_base_torque', 'gripper_positions', 'joint_efforts', 
                                'joint_positions', 'joint_velocities', 'measured_force', 'measured_torque', 'pose', 'velocity']
    
    raw_folder_path = 'data/raw_high_level_actions'
    available_actions = [d for d in os.listdir(raw_folder_path) if os.path.isdir(f'{raw_folder_path}/{d}')]
    for action in available_actions:
        action_path = f'{raw_folder_path}/{action}'
        objects = [o for o in os.listdir(action_path) if os.path.isdir(f'{action_path}/{o}')]
        for obj in objects:
            object_path = f'{action_path}/{obj}'
            available_files = [f for f in os.listdir(object_path) if f.endswith('.npy')]
            for file_name in available_files:
                # Get raw data
                file_path = f'{object_path}/{file_name}'
                high_level_action_dict_raw = np.load(file_path, allow_pickle=True).item()

                modality_files_raw = {}
                for sensor in robot_state_sensor_names:
                    sensor_values = high_level_action_dict_raw[sensor][0]
                    timestamps = high_level_action_dict_raw[sensor][1]
                    modality_files_raw[sensor] = (sensor_values, timestamps)

                # Get interpolated data
                processed_folder_path = 'data/processed_high_level_actions'
                file_path_interpolated = f'{processed_folder_path}/{action}/{obj}/{file_name}'
                if not os.path.exists(file_path_interpolated):
                    print(f"Interpolated file not found: {file_path_interpolated}, skipping...")
                    continue
                high_level_action_dict_interpolated = np.load(file_path_interpolated, allow_pickle=True).item()
                
                modality_files_interpolated = {}
                for sensor in robot_state_sensor_names:
                    sensor_values = high_level_action_dict_interpolated[sensor][0]
                    timestamps = high_level_action_dict_interpolated[sensor][1]
                    modality_files_interpolated[sensor] = (sensor_values, timestamps)

                # Plot comparisons
                sensors_to_plot = ['pose']
                data_plots_dir = f'data/plots/{action}/{obj}/{file_name.replace(".npy", "")}'
                os.makedirs(data_plots_dir, exist_ok=True)
                for sensor in sensors_to_plot:
                    for dim in range(modality_files_interpolated[sensor][0].shape[1]):
                        modality_name = f"{sensor}_{dim}"
                        
                        # Original data
                        sensor_values_raw, timestamps_raw = modality_files_raw[sensor]
                        sensor_values_raw_dim = sensor_values_raw[:, dim]
                        
                        # Interpolated data
                        sensor_values_interpolated, timestamps_interpolated = modality_files_interpolated[sensor]
                        sensor_values_interpolated_dim = sensor_values_interpolated[:, dim]

                        if np.isnan(sensor_values_interpolated_dim).any():
                            print(f"There are NaN values in {modality_name}, skipping plot.")
                            print(sensor_values_interpolated_dim)
                            continue
                        
                        plot_interpolated_values(
                            timestamps_raw,
                            sensor_values_raw_dim,
                            timestamps_interpolated,
                            sensor_values_interpolated_dim,
                            modality_name,
                            data_plots_dir
                        )
