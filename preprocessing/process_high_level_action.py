import numpy as np
from scipy.interpolate import interp1d

def interpolate_sensor_data(sensor_values, timestamps, target_timestamps, kind='cubic'):
    """
    Interpolate sensor data to match target timestamps using spline interpolation.
    
    Args:
        sensor_values: numpy array of shape (n_samples, n_features)
        timestamps: numpy array of shape (n_samples,) - original timestamps
        target_timestamps: numpy array of shape (m_samples,) - desired timestamps
        kind: interpolation type ('linear', 'cubic', 'quadratic', etc.)
    
    Returns:
        interpolated_values: numpy array of shape (m_samples, n_features)
    """
    # Handle 1D case (single feature)
    if sensor_values.ndim == 1:
        sensor_values = sensor_values.reshape(-1, 1)
    
    n_features = sensor_values.shape[1]
    interpolated_values = np.zeros((len(target_timestamps), n_features))
    
    # Interpolate each feature separately
    for i in range(n_features):
        interpolator = interp1d(timestamps, sensor_values[:, i], 
                               kind=kind, 
                               bounds_error=False, 
                               fill_value='extrapolate')
        interpolated_values[:, i] = interpolator(target_timestamps)
    
    return interpolated_values.squeeze()

def synchronize_multiple_modalities(modality_files, target_n_samples=100):
    """
    Synchronize multiple sensor modalities to a common number of samples.
    
    Args:
        modality_files: dict with keys as modality names and values as tuples 
                       (sensor_values, timestamps)
        target_n_samples: target number of samples for all trajectories
    
    Returns:
        synchronized_data: dict with interpolated sensor values for each modality
        common_timestamps: the common timestamp array
    """
    # Load all data
    modalities = {}
    min_time = float('inf')
    max_time = float('-inf')
    
    for modality_name, (values, timestamps) in modality_files.items():
        modalities[modality_name] = (values, timestamps)
        
        # Track time range
        min_time = min(min_time, timestamps.min())
        max_time = max(max_time, timestamps.max())
    
    # Create common timestamp array with fixed number of samples
    common_timestamps = np.linspace(min_time, max_time, target_n_samples)
    
    # Interpolate each modality
    synchronized_data = {}
    for modality_name, (values, timestamps) in modalities.items():
        print(f"Interpolating {modality_name}...")
        synchronized_data[modality_name] = interpolate_sensor_data(
            values, timestamps, common_timestamps, kind='cubic'
        )
    
    return synchronized_data, common_timestamps


if __name__ == '__main__':
    high_level_action_dict = np.load('data/raw_high_level_actions/2025-01-09-13-59-54_high_level_action_31_robot_state.npy', allow_pickle=True).item()
    timestamps_dict = np.load('data/raw_high_level_actions/2025-01-09-13-59-54_high_level_action_31_timestamps.npy', allow_pickle=True).item()
    
    robot_state_sensor_names = ['compensated_base_force', 'compensated_base_torque', 'gripper_positions', 'joint_efforts', 'joint_positions', 'joint_velocities', 'measured_force', 'measured_torque', 'pose', 'velocity']
    
    modality_files = {}
    for sensor in robot_state_sensor_names:
        sensor_values = high_level_action_dict[sensor]
        timestamps = timestamps_dict[sensor]
        modality_files[sensor] = (sensor_values, timestamps)
        
    synchronized_data, common_timestamps = synchronize_multiple_modalities(modality_files, target_n_samples=1000)
    
    print("Synchronized Data Shapes:")
    for sensor in robot_state_sensor_names:
        print(f"{sensor}: {synchronized_data[sensor].shape}")
    