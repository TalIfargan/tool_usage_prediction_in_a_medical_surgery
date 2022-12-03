import os.path
from pathlib import Path
import cv2
import numpy as np
import bbox_visualizer as bbv
import argparse

## https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/      ## basic opencv tutorial
## https://github.com/shoumikchow/bbox-visualizer  ## bbox_visualizer git with examples


def save_all_images(save_path, video_path):
    cap = cv2.VideoCapture(video_path)
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
        exit()
    # making sure save_path exists for the images to be saved
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # Read until video is completed, and save each image
    i = 0
    while cap.isOpened():
        i += 1
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f'{save_path}/{str(i).zfill(8)}.jpg', frame)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def extract_labels(labels_file):
    with open(labels_file) as file:
        labels = [(line.rstrip().split()[0], line.rstrip().split()[-1]) for line in file]
    left_labels = [label for label in labels if int(label[0])%2]
    right_labels = [label for label in labels if not int(label[0])%2]
    left_label, right_label = None, None
    if left_labels:
        left_label = max(left_labels, key=lambda t: t[1])[0]
    if right_labels:
        right_label = max(right_labels, key=lambda t: t[1])[0]
    return left_label, right_label


def predict_tool(history): #TODO
    pass


def record_tool(save_path, tool, start_time, end_time):
    # with open(txt_path + '.txt', 'a') as f:
    #     f.write(('%g ' * len(line)).rstrip() % line + '\n')
    pass


def predict_tool_usage(labels_path, output_path):
    save_path_left = os.path.join(output_path, 'left', 'predictions.txt')
    save_path_right = os.path.join(output_path, 'right', 'predictions.txt')
    pred_files = sorted(os.listdir(labels_path))
    first_file = os.path.join(labels_path, pred_files[0])
    start_frame_left = 0
    start_frame_right = 0
    current_tool_left, current_tool_right = extract_labels(first_file)
    smoothed_tool_left, smoothed_tool_right = current_tool_left, current_tool_right
    left_history = [current_tool_left]
    right_history = [current_tool_right]
    for i, file in enumerate(pred_files[1:]):
        current_tool_left, current_tool_right = extract_labels(os.path.join(labels_path, file))
        left_history = left_history[-9:] + [current_tool_left]
        right_history = right_history[-9:] + [current_tool_right]
        pred_left = predict_tool(left_history)
        pred_right = predict_tool(right_history)
        if pred_left != smoothed_tool_left or i == len(pred_files)-2:
            record_tool(save_path_left, smoothed_tool_left, start_frame_left, i)  # i because enumeration is different from indexing
            smoothed_tool_left = pred_left
            start_frame_left = i + 1
        if pred_right != smoothed_tool_right or i == len(pred_files)-2:
            record_tool(save_path_right, smoothed_tool_right, start_frame_right, i)
            smoothed_tool_right = pred_right
            start_frame_right = i+1


def run_inference(video_args):
    # making sure desired paths exist
    video_name = Path(video_args.video_path).stem
    if video_args.video_frames_path not in os.listdir():
        os.mkdir(video_args.video_frames_path)
    save_path = os.path.join(video_args.video_frames_path, video_name)

    # saving all frames
    if video_args.save_images:
        # making sure the frames are not already saved
        save_all = 'y'
        if os.listdir(save_path):
            print('frame save path is not empty. Are you sure you wish to save all images? (y/n)')
            save_all = input()
        if save_all.lower() == 'y':
            print(f'reading and saving all the frames of {video_name} video')
            save_all_images(save_path, video_args.video_path)

    # running inference for all frames
    if video_args.infer_images:
        if video_name not in os.listdir('model_output'):
            os.mkdir(os.path.join('model_output', video_name))
        # making sure the inference is needed
        infer_all = 'y'
        if os.listdir(os.path.join('model_output', video_name)):
            print('inference path is not empty. Are you sure you wish to run inference for all images? (y/n)')
            infer_all = input()
        if infer_all.lower() == 'y':
            print(f'doing inference for all frames of {video_name} video')
            os.system(f'python predict.py --source {save_path} --weights weights/best.pt --nosave --save-txt --project model_output --name {video_name} --save-conf --smooth_tool')

    # running tool usage prediction using model's outputs
    if video_args.predict_tools:
        if 'tool_usage_prediction' not in os.listdir(os.path.join('model_output', video_name)):
            os.mkdir(os.path.join('model_output', video_name, 'tool_usage_prediction'))
        predict_tool_usage(os.path.join('model_output', video_name, 'labels'),
                           os.path.join('model_output', video_name, 'tool_usage_prediction'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', help='path to desired video for inference')
    parser.add_argument('--video_frames_path', default='video_frames', help='path for the video frames to be saved in. will contain another sub-directory for the specific video')
    parser.add_argument('--save_images', action='store_true', help='saving all images or not')
    parser.add_argument('--infer_images', action='store_true', help='doing inference for all images or not')
    parser.add_argument('--predict_tools', action='store_true', help='produce a tool prediction file or not')
    parser.add_argument('--smooth_tool_prediction', action='store_true', help='use smoothing in tool usage prediction')
    video_args = parser.parse_args()
    run_inference(video_args)