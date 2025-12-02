from REASSEMBLE.io import load_h5_file
import os
import io
import shutil
import cv2
import numpy as np


def save_video_segment(video_data, timestamps, start_time, end_time, output_path):
    """
    Extract and save a segment of video based on timestamp range.
    
    Args:
        video_data (bytes): Binary video data from H5 file.
        timestamps (np.ndarray): Array of frame timestamps.
        start_time (float): Start timestamp for the segment.
        end_time (float): End timestamp for the segment.
        output_path (str): Path where the video segment should be saved.
    """
    # Write full video to temporary file
    temp_input = "temp_full_video.mp4"
    with open(temp_input, "wb") as f:
        binary_stream = io.BytesIO(video_data)
        shutil.copyfileobj(binary_stream, f)
    
    # Find frame indices within the time range
    frame_indices = np.where((timestamps >= start_time) & (timestamps <= end_time))[0]
    
    if len(frame_indices) == 0:
        print(f"Warning: No frames found in the time range [{start_time}, {end_time}]")
        os.remove(temp_input)
        return
    
    # Open the video file
    cap = cv2.VideoCapture(temp_input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Get frame dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Extract frames
    current_frame = 0
    frame_idx_set = set(frame_indices)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame in frame_idx_set:
            out.write(frame)
        
        current_frame += 1
    
    # Release resources
    cap.release()
    out.release()
    
    # Clean up temporary file
    os.remove(temp_input)
    print(f"Saved video segment to {output_path}")


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
    for sensor in robot_state_sensor_names:
        indexes = np.where((data['timestamps'][sensor] >= start) & (data['timestamps'][sensor] <= end))
        sensor_data = data['robot_state'][sensor][indexes]
        robot_state_sensor_values[sensor] = sensor_data
        print(f"{sensor}: {sensor_data.shape}")
        
        
    np.save(f'data/raw_high_level_actions/{file_name}_high_level_action_{high_level_action_KEY}_robot_state.npy', robot_state_sensor_values)
    

    # Save the video segment
    video_keys = ['hama1', 'hama2', 'hand']
    high_level_action_name = "pick_small_peg"
    output_dir = f'data/videos/{file_name}_{high_level_action_name}'
    
    for video_key in video_keys:
        video_data = data[video_key]
        video_timestamps = data['timestamps'][video_key]
        output_path = f'{output_dir}/{video_key}.mp4'
        save_video_segment(video_data, video_timestamps, start, end, output_path)
        print(f"Saved {video_key} video segment")
    