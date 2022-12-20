import argparse
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import editdistance


def tool_by_frame(tool_records):
    tool_by_frame = []
    for record in tool_records:
        tool_by_frame += [record[2]] * (int(record[1]) - int(record[0]) + 1)
    return tool_by_frame

def extract_lines(tool_file):
    with open(tool_file) as file:
        tools = [line.rstrip().split() for line in file]
        tools = [(line[0].strip(), line[1].strip(), line[2].strip()) for line in tools]
        return tools

def evaluate_tool_usage(pred_file_left, gt_file_left, pred_file_right, gt_file_right, video_name, smoothing_method):
    tools_pred_left, tools_gt_left, tools_pred_right, tools_gt_right =\
        extract_lines(pred_file_left), extract_lines(gt_file_left), extract_lines(pred_file_right), extract_lines(gt_file_right)
    edit_string_left_pred = ''.join([tool[-1] for tool in tools_pred_left]).replace('T','')
    edit_string_left_gt = ''.join([tool[-1] for tool in tools_gt_left]).replace('T','')
    edit_string_right_pred = ''.join([tool[-1] for tool in tools_pred_right]).replace('T','')
    edit_string_right_gt = ''.join([tool[-1] for tool in tools_gt_right]).replace('T','')
    pred_left, gt_left, pred_right, gt_right = \
        tool_by_frame(tools_pred_left), tool_by_frame(tools_gt_left), tool_by_frame(tools_pred_right), tool_by_frame(tools_gt_right)
    if len(gt_left) > len(pred_left):
        gt_left = gt_left[:len(pred_left)]
    if len(pred_left) > len(gt_left):
        pred_left = pred_left[:len(gt_left)]
    if len(gt_right) > len(pred_right):
        gt_right = gt_right[:len(pred_right)]
    if len(pred_right) > len(gt_right):
        pred_right = pred_right[:len(gt_right)]
    pred = np.array(pred_left + pred_right)
    gt = np.array(gt_left + gt_right)
    T0_pred = pred.copy()
    T0_pred[T0_pred != 'T0'] = 0
    T0_gt = gt.copy()
    T0_gt[T0_gt != 'T0'] = 0

    T1_pred = pred.copy()
    T1_pred[T1_pred != 'T1'] = 0
    T1_gt = gt.copy()
    T1_gt[T1_gt != 'T1'] = 0

    T2_pred = pred.copy()
    T2_pred[T2_pred != 'T2'] = 0
    T2_gt = gt.copy()
    T2_gt[T2_gt != 'T2'] = 0

    T3_pred = pred.copy()
    T3_pred[T3_pred != 'T3'] = 0
    T3_gt = gt.copy()
    T3_gt[T3_gt != 'T3'] = 0

    recall_T0 = recall_score(T0_gt, T0_pred, pos_label='T0', average='binary')
    recall_T1 = recall_score(T1_gt, T1_pred, pos_label='T1', average='binary')
    recall_T2 = recall_score(T2_gt, T2_pred, pos_label='T2', average='binary')
    recall_T3 = recall_score(T3_gt, T3_pred, pos_label='T3', average='binary')
    recall_micro = recall_score(gt, pred, average='micro')
    precision_T0 = precision_score(T0_gt, T0_pred, pos_label='T0', average='binary')
    precision_T1 = precision_score(T1_gt, T1_pred, pos_label='T1', average='binary')
    precision_T2 = precision_score(T2_gt, T2_pred, pos_label='T2', average='binary')
    precision_T3 = precision_score(T3_gt, T3_pred, pos_label='T3', average='binary')
    precision_micro = precision_score(gt, pred, average='micro')
    f1_T0 = f1_score(T0_gt, T0_pred, pos_label='T0', average='binary')
    f1_T1 = f1_score(T1_gt, T1_pred, pos_label='T1', average='binary')
    f1_T2 = f1_score(T2_gt, T2_pred, pos_label='T2', average='binary')
    f1_T3 = f1_score(T3_gt, T3_pred, pos_label='T3', average='binary')
    f1_macro = f1_score(gt, pred, average='macro')
    f1_micro = f1_score(gt, pred, average='micro')
    overall_accuracy = accuracy_score(gt, pred)
    recall_results_df = pd.DataFrame([recall_T0, recall_T1, recall_T2, recall_T3, recall_micro])
    recall_results_df.index = ['recall_T0', 'recall_T1', 'recall_T2', 'recall_T3', 'recall_micro']
    precision_results_df = pd.DataFrame([precision_T0, precision_T1, precision_T2, precision_T3, precision_micro])
    precision_results_df.index = ['precision_T0', 'precision_T1', 'precision_T2', 'precision_T3', 'precision_micro']
    f1_results_df = pd.DataFrame([f1_T0, f1_T1, f1_T2, f1_T3, f1_macro, f1_micro])
    f1_results_df.index = ['f1_T0', 'f1_T1', 'f1_T2', 'f1_T3', 'f1_macro', 'f1_micro']
    general_results_df = pd.DataFrame([overall_accuracy, f1_macro, precision_micro, recall_micro])
    general_results_df.index = ['overall_accuracy', 'f1_macro', 'precision_micro', 'recall_micro']
    edit_score_left = 1 - (editdistance.eval(edit_string_left_pred, edit_string_left_gt)/max(len(edit_string_left_pred),len(edit_string_left_gt)))
    edit_score_right = 1 - (editdistance.eval(edit_string_right_pred, edit_string_right_gt)/max(len(edit_string_right_pred),len(edit_string_right_gt)))
    edit_score_mean = (edit_score_left+edit_score_right)/2
    print(recall_results_df)
    print(precision_results_df)
    print(f1_results_df)
    print(general_results_df)
    print('edit_score_left = ', edit_score_left)
    print('edit_score_right = ', edit_score_right)
    print('edit_score_mean = ', edit_score_mean)

    with open(f'analysis/{video_name}_{smoothing_method}.txt', 'a') as f:
        f.write(f'recall_T0 {recall_T0}\n')
        f.write(f'recall_T1 {recall_T1}\n')
        f.write(f'recall_T2 {recall_T2}\n')
        f.write(f'recall_T3 {recall_T3}\n')

        f.write(f'precision_T0 {precision_T0}\n')
        f.write(f'precision_T1 {precision_T1}\n')
        f.write(f'precision_T2 {precision_T2}\n')
        f.write(f'precision_T3 {precision_T3}\n')

        f.write(f'f1_T0 {f1_T0}\n')
        f.write(f'f1_T1 {f1_T1}\n')
        f.write(f'f1_T2 {f1_T2}\n')
        f.write(f'f1_T3 {f1_T3}\n')

        f.write(f'f1_macro {f1_macro}\n')
        f.write(f'overall_accuracy {overall_accuracy}\n')
        f.write(f'edit_score_mean {edit_score_mean}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_name')
    parser.add_argument('--smoothing_method')
    parser.add_argument('--pred_file_path_left', help='path to the .txt file containing the tool usage predictions - left')
    parser.add_argument('--pred_file_path_right', help='path to the .txt file containing the tool usage predictions - right')
    parser.add_argument('--gt_file_path_left', help='path to the .txt file containing the tool usage ground truth - left')
    parser.add_argument('--gt_file_path_right', help='path to the .txt file containing the tool usage ground truth - right')
    eval_args = parser.parse_args()
    evaluate_tool_usage(eval_args.pred_file_path_left, eval_args.gt_file_path_left, eval_args.pred_file_path_right,
                        eval_args.gt_file_path_right, eval_args.video_name, eval_args.smoothing_method)