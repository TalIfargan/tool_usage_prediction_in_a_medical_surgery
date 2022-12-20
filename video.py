import os.path
from pathlib import Path
import cv2
import numpy as np
import bbox_visualizer as bbv
import argparse
import glob

HISTORY_LENGTH = 30

T = {'mean': [1]*HISTORY_LENGTH,
     'exp': np.exp(-0.1*np.arange(HISTORY_LENGTH)),
     'none': [1]+[0]*(HISTORY_LENGTH-1)}


TOOL_NAMING = {'0': 'Scissors',
               '1': 'Scissors',
               '2': 'Needle_driver',
               '3': 'Needle_driver',
               '4': 'Forceps',
               '5': 'Forceps',
               '6': 'Empty',
               '7': 'Empty'}

NAME_FROM_TOOL = {'T3': 'Scissors',
                  'T1': 'Needle_driver',
                  'T2': 'Forceps',
                  'T0': 'Empty'}

TOOL_CONVERTER = {'0': 'T3',
                  '1': 'T3',
                  '2': 'T1',
                  '3': 'T1',
                  '4': 'T2',
                  '5': 'T2',
                  '6': 'T0',
                  '7': 'T0'}

LABEL_DICT = {'0': 0,
              '1': 0,
              '2': 0,
              '3': 0,
              '4': 0,
              '5': 0,
              '6': 0,
              '7': 0}


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


def extract_gt_file(gt_file):
    with open(gt_file) as file:
        usage = [line.rstrip().split() for line in file]
        usage = [(line[0].strip(), line[1].strip(), line[2].strip()) for line in usage]
        return usage


def extract_labels_and_bbox(labels_file):
    with open(labels_file) as file:
        labels = [line.rstrip().split() for line in file]
    left_labels = [label for label in labels if int(label[0]) % 2]
    right_labels = [label for label in labels if not int(label[0]) % 2]
    left_label, right_label = '', ''
    bbox_left, bbox_right = [], []
    if left_labels:
        left = max(left_labels, key=lambda t: t[-1])
        left_label = left[0]
        bbox_left = [left[1], left[2], left[3], left[4]]
    if right_labels:
        right = max(right_labels, key=lambda t: t[-1])
        right_label = right[0]
        bbox_right = [right[1], right[2], right[3], right[4]]
    return left_label.strip(), right_label.strip(), bbox_left, bbox_right


def predict_tool(history, smoothing_method):
    label_dict = LABEL_DICT.copy()
    for i, label in enumerate(history[::-1]):
        label_dict[label] += T[smoothing_method][i]
    return max(label_dict, key=label_dict.get)


def record_tool(save_path, tool, start_time, end_time):
    with open(save_path, 'a') as f:
        f.write(f'{start_time} {end_time} {TOOL_CONVERTER[tool]}\n')


def get_gt_tools(gt_left, gt_right, frame_idx):
    left_gt_label, right_gt_label = None, None
    for record in gt_left:
        if int(record[0]) <= frame_idx <= int(record[1]):
            left_gt_label = record[2]
            break
    for record in gt_right:
        if int(record[0]) <= frame_idx <= int(record[1]):
            right_gt_label = record[2]
            break
    return NAME_FROM_TOOL[left_gt_label], NAME_FROM_TOOL[right_gt_label]


def write_frame(video_name, file, bbox_left, bbox_right, label_left, label_right, gt_left, gt_right, frame_idx):
    frame = cv2.imread(os.path.join('video_frames', video_name, Path(file).stem + '.jpg'))
    left_gt_label, right_gt_label = get_gt_tools(gt_left, gt_right, frame_idx)
    manually_up = np.array([0, -70, 0, 0])
    manually_left = np.array([-30, 0, 0, 0])
    manually_right = np.array([30, 0, 0, 0])
    if bbox_left:
        bbox_left = [float(num) for num in bbox_left]
        bbox_left = [int((bbox_left[0]-0.5*bbox_left[2])*frame.shape[1]), int((bbox_left[1]-0.5*bbox_left[3])*frame.shape[0]),
                     int((bbox_left[0]+0.5*bbox_left[2])*frame.shape[1]), int((bbox_left[1]+0.5*bbox_left[3])*frame.shape[0])]
        frame = bbv.draw_rectangle(frame, bbox_left, bbox_color=(255, 0, 0))
        frame = bbv.add_label(frame, f'L_{TOOL_NAMING[label_left]}', bbox_left, top=True)
        frame = bbv.add_label(frame, 'GT: ' + left_gt_label, bbox_left+manually_up, top=False, text_color=(255, 0, 0))
        #frame = bbv.draw_rectangle(frame, bbox_left+manually_up, bbox_color=(0,0,0))
    if bbox_right:
        bbox_right = [float(num) for num in bbox_right]
        bbox_right = [int((bbox_right[0] - 0.5 * bbox_right[2]) * frame.shape[1]), int((bbox_right[1] - 0.5 * bbox_right[3]) * frame.shape[0]),
                     int((bbox_right[0] + 0.5 * bbox_right[2]) * frame.shape[1]), int((bbox_right[1] + 0.5 * bbox_right[3]) * frame.shape[0])]
        frame = bbv.draw_rectangle(frame, bbox_right, bbox_color=(0, 255, 0))
        frame = bbv.add_label(frame, f'R_{TOOL_NAMING[label_right]}', bbox_right, top=True)
        frame = bbv.add_label(frame, 'GT: ' + right_gt_label, bbox_right+manually_up, top=False, text_color=(0,255,0))
    cv2.imwrite(os.path.join('model_output', video_name, 'labeled_images', Path(file).stem + '.jpg'), frame)


def image_seq_to_video(imgs_path, output_path='./video.mp4', fps=30.0):
    output = output_path
    img_array = []
    for filename in sorted(glob.glob(os.path.join(imgs_path, '*.jpg'))):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        # img = cv2.resize(img, (width // 2, height // 2))
        img = cv2.resize(img, (width, height))
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    print(size)
    print("writing video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, fps, size)
    # out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print("saved video @ ", output)


def predict_tool_usage(labels_path, output_path, write_video, video_name, left_gt_path, right_gt_path, smoothing_method):
    if 'left' not in os.listdir(output_path):
        os.mkdir(os.path.join(output_path, 'left'))
    if 'right' not in os.listdir(output_path):
        os.mkdir(os.path.join(output_path, 'right'))
    save_path_left = os.path.join(output_path, 'left', f'predictions_{smoothing_method}.txt')
    save_path_right = os.path.join(output_path, 'right', f'predictions_{smoothing_method}.txt')
    pred_files = sorted(os.listdir(labels_path))
    gt_left, gt_right = extract_gt_file(left_gt_path), extract_gt_file(right_gt_path)
    first_file = os.path.join(labels_path, pred_files[0])
    start_frame_left = 0
    start_frame_right = 0
    current_tool_left, current_tool_right, bbox_left, bbox_right = extract_labels_and_bbox(first_file)
    smoothed_tool_left, smoothed_tool_right = current_tool_left, current_tool_right
    if write_video:
        write_frame(video_name, pred_files[0], bbox_left, bbox_right, current_tool_left, smoothed_tool_left, gt_left, gt_right, 0)
    left_history = [current_tool_left]
    right_history = [current_tool_right]
    for i, file in enumerate(pred_files[1:]):
        current_tool_left, current_tool_right, bbox_left, bbox_right = extract_labels_and_bbox(os.path.join(labels_path, file))
        # Note - if no prediction for hand - ignore and predict previous one
        left_history = left_history[-(HISTORY_LENGTH)+1:] + [current_tool_left] if current_tool_left else  left_history
        right_history = right_history[-(HISTORY_LENGTH)+1:] + [current_tool_right] if current_tool_right else  right_history
        pred_left = predict_tool(left_history, smoothing_method)
        pred_right = predict_tool(right_history, smoothing_method)
        if write_video:
            write_frame(video_name, file, bbox_left, bbox_right, pred_left, pred_right, gt_left, gt_right, i+1)
        if i == len(pred_files)-2:
            record_tool(save_path_left, smoothed_tool_left, start_frame_left, i+1) #unique case
            record_tool(save_path_right, smoothed_tool_right, start_frame_right, i+1)
        if pred_left != smoothed_tool_left:
            record_tool(save_path_left, smoothed_tool_left, start_frame_left, i)  # i because enumeration is different from indexing
            smoothed_tool_left = pred_left
            start_frame_left = i + 1
        if pred_right != smoothed_tool_right:
            record_tool(save_path_right, smoothed_tool_right, start_frame_right, i)
            smoothed_tool_right = pred_right
            start_frame_right = i+1
    # write final video
    if write_video:
        image_seq_to_video(os.path.join('model_output', video_name, 'labeled_images'),
                           output_path=os.path.join('model_output', video_name, 'labeled_video.mp4'), fps=30.0)

def run_inference(video_args):
    # making sure desired paths exist
    video_name = Path(video_args.video_path).stem
    if video_args.video_frames_path not in os.listdir():
        os.mkdir(video_args.video_frames_path)
    if video_name not in os.listdir(os.path.join(video_args.video_frames_path)):
        os.mkdir(os.path.join(video_args.video_frames_path, video_name))
    save_path = os.path.join(video_args.video_frames_path, video_name)

    # saving all frames
    if video_args.save_images:
        # making sure the frames are not already saved
        save_all = 'y'
        if os.listdir(save_path):
            print('frame save path may not be empty. Are you sure you wish to save all images? (y/n)')
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
            os.system(f'python predict.py --source {save_path} --weights weights/best.pt --nosave --save-txt --project model_output --name {video_name} --save-conf --exist-ok')

    # running tool usage prediction using model's outputs
    if video_args.predict_tools:
        if 'tool_usage_prediction' not in os.listdir(os.path.join('model_output', video_name)):
            os.mkdir(os.path.join('model_output', video_name, 'tool_usage_prediction'))
        if 'labeled_images' not in os.listdir(os.path.join('model_output', video_name)):
            os.mkdir(os.path.join('model_output', video_name, 'labeled_images'))
        if 'labels' not in os.listdir(os.path.join('model_output', video_name)):
            os.mkdir(os.path.join('model_output', video_name, 'labels'))
        predict_tool_usage(os.path.join('model_output', video_name, 'labels'),
                           os.path.join('model_output', video_name, 'tool_usage_prediction'),
                           video_args.write_video, video_name, video_args.left_gt_path,
                           video_args.right_gt_path, video_args.smoothing_method)

    # saving updated video



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', help='path to desired video for inference')
    parser.add_argument('--video_frames_path', default='video_frames', help='path for the video frames to be saved in. will contain another sub-directory for the specific video')
    parser.add_argument('--save_images', action='store_true', help='saving all images or not')
    parser.add_argument('--infer_images', action='store_true', help='doing inference for all images or not')
    parser.add_argument('--predict_tools', action='store_true', help='produce a tool prediction file or not')
    parser.add_argument('--write_video', action='store_true', help='produce a video with the predictions')
    parser.add_argument('--left_gt_path', help='path to gt tool usage file of the left hand')
    parser.add_argument('--right_gt_path', help='path to gt tool usage file of the right hand')
    parser.add_argument('--smoothing_method', default='none', help='mean, exp or none, will be as a part of the predictions txt file')
    video_args = parser.parse_args()
    run_inference(video_args)
