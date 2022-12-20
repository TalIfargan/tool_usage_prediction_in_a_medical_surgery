import os
import numpy as np

def create_dict(file_name):
    # Read in file
    lines = []
    with open(f'analysis/{file_name}', 'r') as f:
        lines = f.readlines()
    # Split out and drop empty rows
    strip_list = [line.replace('\n','').split(' ') for line in lines if line != '\n']
    d = dict()
    for strip in strip_list:
        d[strip[0]] = float(strip[1])
    return d


result = {file_name: {} for file_name in os.listdir('analysis')}

for file_name in result.keys():
    result[file_name] = create_dict(file_name)

# recall
recall_TO_none = np.mean([result[file]['recall_T0'] for file in result.keys() if 'none' in file])
recall_T1_none = np.mean([result[file]['recall_T1'] for file in result.keys() if 'none' in file])
recall_T2_none = np.mean([result[file]['recall_T2'] for file in result.keys() if 'none' in file])
recall_T3_none = np.mean([result[file]['recall_T3'] for file in result.keys() if 'none' in file])
recall_none = [recall_TO_none, recall_T1_none, recall_T2_none, recall_T3_none]
print('recall_none = ', recall_none)

recall_TO_mean = np.mean([result[file]['recall_T0'] for file in result.keys() if 'mean' in file])
recall_T1_mean = np.mean([result[file]['recall_T1'] for file in result.keys() if 'mean' in file])
recall_T2_mean = np.mean([result[file]['recall_T2'] for file in result.keys() if 'mean' in file])
recall_T3_mean = np.mean([result[file]['recall_T3'] for file in result.keys() if 'mean' in file])
recall_mean = [recall_TO_mean, recall_T1_mean, recall_T2_mean, recall_T3_mean]
print('recall_mean = ', recall_mean)

recall_TO_exp = np.mean([result[file]['recall_T0'] for file in result.keys() if 'exp' in file])
recall_T1_exp = np.mean([result[file]['recall_T1'] for file in result.keys() if 'exp' in file])
recall_T2_exp = np.mean([result[file]['recall_T2'] for file in result.keys() if 'exp' in file])
recall_T3_exp = np.mean([result[file]['recall_T3'] for file in result.keys() if 'exp' in file])
recall_exp = [recall_TO_exp, recall_T1_exp, recall_T2_exp, recall_T3_exp]
print('recall_exp = ', recall_exp)

# precision
precision_TO_none = np.mean([result[file]['precision_T0'] for file in result.keys() if 'none' in file])
precision_T1_none = np.mean([result[file]['precision_T1'] for file in result.keys() if 'none' in file])
precision_T2_none = np.mean([result[file]['precision_T2'] for file in result.keys() if 'none' in file])
precision_T3_none = np.mean([result[file]['precision_T3'] for file in result.keys() if 'none' in file])
precision_none = [precision_TO_none, precision_T1_none, precision_T2_none, precision_T3_none]
print('precision_none = ', precision_none)

precision_TO_mean = np.mean([result[file]['precision_T0'] for file in result.keys() if 'mean' in file])
precision_T1_mean = np.mean([result[file]['precision_T1'] for file in result.keys() if 'mean' in file])
precision_T2_mean = np.mean([result[file]['precision_T2'] for file in result.keys() if 'mean' in file])
precision_T3_mean = np.mean([result[file]['precision_T3'] for file in result.keys() if 'mean' in file])
precision_mean = [precision_TO_mean, precision_T1_mean, precision_T2_mean, precision_T3_mean]
print('precision_mean = ', precision_mean)

precision_TO_exp = np.mean([result[file]['precision_T0'] for file in result.keys() if 'exp' in file])
precision_T1_exp = np.mean([result[file]['precision_T1'] for file in result.keys() if 'exp' in file])
precision_T2_exp = np.mean([result[file]['precision_T2'] for file in result.keys() if 'exp' in file])
precision_T3_exp = np.mean([result[file]['precision_T3'] for file in result.keys() if 'exp' in file])
precision_exp = [precision_TO_exp, precision_T1_exp, precision_T2_exp, precision_T3_exp]
print('precision_exp = ', precision_exp)

# f1
f1_TO_none = np.mean([result[file]['f1_T0'] for file in result.keys() if 'none' in file])
f1_T1_none = np.mean([result[file]['f1_T1'] for file in result.keys() if 'none' in file])
f1_T2_none = np.mean([result[file]['f1_T2'] for file in result.keys() if 'none' in file])
f1_T3_none = np.mean([result[file]['f1_T3'] for file in result.keys() if 'none' in file])
f1_none = [f1_TO_none, f1_T1_none, f1_T2_none, f1_T3_none]
print('f1_none = ', f1_none)

f1_TO_mean = np.mean([result[file]['f1_T0'] for file in result.keys() if 'mean' in file])
f1_T1_mean = np.mean([result[file]['f1_T1'] for file in result.keys() if 'mean' in file])
f1_T2_mean = np.mean([result[file]['f1_T2'] for file in result.keys() if 'mean' in file])
f1_T3_mean = np.mean([result[file]['f1_T3'] for file in result.keys() if 'mean' in file])
f1_mean = [f1_TO_mean, f1_T1_mean, f1_T2_mean, f1_T3_mean]
print('f1_mean = ', f1_mean)

f1_TO_exp = np.mean([result[file]['f1_T0'] for file in result.keys() if 'exp' in file])
f1_T1_exp = np.mean([result[file]['f1_T1'] for file in result.keys() if 'exp' in file])
f1_T2_exp = np.mean([result[file]['f1_T2'] for file in result.keys() if 'exp' in file])
f1_T3_exp = np.mean([result[file]['f1_T3'] for file in result.keys() if 'exp' in file])
f1_exp = [f1_TO_exp, f1_T1_exp, f1_T2_exp, f1_T3_exp]
print('f1_exp = ', f1_exp)

# f1 macro
macro_f1_none = np.mean([result[file]['f1_macro'] for file in result.keys() if 'none' in file])
macro_f1_mean = np.mean([result[file]['f1_macro'] for file in result.keys() if 'mean' in file])
macro_f1_exp = np.mean([result[file]['f1_macro'] for file in result.keys() if 'exp' in file])
f1_macro = [macro_f1_none, macro_f1_mean, macro_f1_exp]
print('f1_macro = ', f1_macro)

# overall accuracy
overall_accuracy_none = np.mean([result[file]['overall_accuracy'] for file in result.keys() if 'none' in file])
overall_accuracy_mean = np.mean([result[file]['overall_accuracy'] for file in result.keys() if 'mean' in file])
overall_accuracy_exp = np.mean([result[file]['overall_accuracy'] for file in result.keys() if 'exp' in file])
overall_accuracy = [overall_accuracy_none, overall_accuracy_mean, overall_accuracy_exp]
print('overall_accuracy = ', overall_accuracy)

# mean edit
mean_edit_none = np.mean([result[file]['edit_score_mean'] for file in result.keys() if 'none' in file])
mean_edit_mean = np.mean([result[file]['edit_score_mean'] for file in result.keys() if 'mean' in file])
mean_edit_exp = np.mean([result[file]['edit_score_mean'] for file in result.keys() if 'exp' in file])
edit_score = [mean_edit_none, mean_edit_mean, mean_edit_exp]
print('edit_score = ', edit_score)
