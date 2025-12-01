from REASSEMBLE.io import load_h5_file
import numpy as np


if __name__ == '__main__':
    file_name = '2025-01-09-13-59-54'
    h5_file_path = f'data/original_reassemble_data/{file_name}.h5'
    data = load_h5_file(h5_file_path, decode=False)

    segments_info = data.get('segments_info', None)
    high_level_action_KEY = '31'
    high_level_action = segments_info.get(high_level_action_KEY)
    start, end = high_level_action.get('start'), high_level_action.get('end')

    robot_state_sensor_names = ['compensated_base_force', 'compensated_base_torque', 'gripper_positions', 'joint_efforts', 'joint_positions', 'joint_velocities', 'measured_force', 'measured_torque', 'pose', 'velocity']
    robot_state_sensor_values = {}
    timestamps_sensor_values = {}
    for sensor in robot_state_sensor_names:
        indexes = np.where((data['timestamps'][sensor] >= start) & (data['timestamps'][sensor] <= end))
        sensor_data = data['robot_state'][sensor][indexes]
        robot_state_sensor_values[sensor] = sensor_data
        timestamps_sensor_values[sensor] = data['timestamps'][sensor][indexes]
        print(f"{sensor}: {sensor_data.shape}")
        
    np.save(f'data/raw_high_level_actions/{file_name}_high_level_action_{high_level_action_KEY}_robot_state.npy', robot_state_sensor_values)
    np.save(f'data/raw_high_level_actions/{file_name}_high_level_action_{high_level_action_KEY}_timestamps.npy', timestamps_sensor_values)