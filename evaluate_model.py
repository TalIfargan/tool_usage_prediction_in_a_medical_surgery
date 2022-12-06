import numpy as np
import tqdm
import os
import pandas as pd

NUM_CLASSES = 8

def calculate_iou(gt_match, prediction):
    '''
    gt_match = [class, x, y, w, h]
    prediction = [class, x, y, w, h]
    '''
    # calculate intersection
    x1 = max(gt_match[1], prediction[1])
    y1 = max(gt_match[2], prediction[2])
    x2 = min(gt_match[1] + gt_match[3], prediction[1] + prediction[3])
    y2 = min(gt_match[2] + gt_match[4], prediction[2] + prediction[4])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    # calculate union
    union = gt_match[3] * gt_match[4] + prediction[3] * prediction[4] - intersection
    # calculate IoU
    iou = intersection / union
    return iou

def calculate_AP(gt, predictions, iou_threshold):
    '''
    calculate average precision for a single image for all NUM_CLASSES classes
    # gt = [class, x, y, w, h]
    # prediction = [class, x, y, w, h]
    '''
    APs = {c: 1 for c in range(NUM_CLASSES)}
    for c in range(NUM_CLASSES):
        # get ground truth bounding box for class c
        gt_c = gt[gt[:, 0].astype(int) == c]
        # get all predictions for class c
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(1, -1)
        predictions_c = predictions[predictions[:, 0].astype(int) == c]

        # if there are no predictions for class c without ground truth, then AP is 1
        if len(gt_c) == 0 and len(predictions_c) == 0:
            continue

        # if there are ground truths, and no predictions then AP is 0
        if len(gt_c) == 0 and len(predictions_c) > 0:
            APs[c] = 0
            continue

        # initialize true positives and false positives for the right size
        tp = np.zeros(predictions_c.shape[0])
        fp = np.zeros(predictions_c.shape[0])
        # loop through predictions
        for i, prediction in enumerate(predictions_c):
            # calculate IoU
            iou = calculate_iou(gt_c[0], prediction)
            # if IoU is higher than threshold, it is a true positive
            if iou > iou_threshold:
                tp[i] = 1
            # if IoU is lower than threshold, it is a false positive
            else:
                fp[i] = 1
        # calculate precision and recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / gt_c.shape[0]
        prec = tp / (fp + tp)
        # calculate average precision
        AP = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            AP = AP + p / 11
        APs[c] = AP
    return APs


# given IoU threshold, evaluate the model
def evaluate_model(gt_path, predictions_path, iou_threshold):
    APs = []
    mAp = []
    for filename in tqdm.tqdm(os.listdir(gt_path)):
        if filename.endswith(".txt"):
            gt = np.loadtxt(os.path.join(gt_path, filename))
            predictions = np.loadtxt(os.path.join(predictions_path, filename))
            AP = calculate_AP(gt, predictions, iou_threshold)
            APs.append(AP)
    results_df = pd.DataFrame(APs)
    results_df.columns = ['AP_' + str(c) for c in range(NUM_CLASSES)]
    results = results_df.mean(axis=0)
    results['mAP'] = results.mean()
    return results


if __name__ == "__main__":
    # evaluate model
    results_25 = evaluate_model("test/labels", "test/predictions", 0.25)
    results_50 = evaluate_model("test/labels", "test/predictions", 0.5)
    results_75 = evaluate_model("test/labels", "test/predictions", 0.75)
    # create dataframe
    results_df = pd.DataFrame([results_25, results_50, results_75])
    results_df.index = ['IoU = 0.25', 'IoU = 0.5', 'IoU = 0.75']
    print(results_df)