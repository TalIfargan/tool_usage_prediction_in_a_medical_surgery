# CVSA_HW1
This git repository can be used to produce and evaluate end to end tool usage prediction in a medical surgery, given a video of it.

## Image Pipeline
Given an image, we can run `predict.py` with the appropriate parameters to produce the following outputs:
* `.txt` file with the `(class, x, y, w, h, conf)` results, this file is created inside a `<name>/labels/` folder 
* Annotated image with the boundings boxes

Both of those will be created using the fine-tuned YOLOV7 we have trained, by loading the best weights from training and running an inference over the given image\frame.
To see the available parameters available use:
```
python predict.py --help
```
from the project directory.
Actually dealing with these parameters is not really neccesery, **just use the example**.

### Example
```
python predict.py --weights weights\best.pt --source path\to\img.jpg --save-txt --save-conf --name NAME-OF-FOLDER
```
Where:
* `path\to\img.jpg` is the path to the image you are trying to annotate
* `NAME-OF-FOLDER` is the name of the folder in which the prediction `.txt` file and the annotated image are saved

## Video Pipeline
Given a video, we can run video.py with the appropriate parameters to produce the following outputs:
* Fine-tuned YOLOV7 outputs per frame.
* Tool usage prediction file per hand (using the model's outputs).
* Labled video - condtaining the final prediction and the ground-truth for each frame.

The run parameters are:
* `video_path`
* `left_gt_path`
* `right_gt_path`
* `smoothing_method` - can be 'none','mean' or 'exp'

The following parameters are Boolean and exist to enable running only some of the modules, especially for debugging:
* `save_images` - whether or not saving the video frames
* `infer_images` - whether or not running the YOLOV7 model one the video frames
* `predict_tools` - whether or not making a final prediction per hand per frame
* `write_video` - whether or not writing the video

### Example
```bash
python video.py --video_path videos/P022_balloon1.wmv --left_gt_path tools_gt/tools_left/P022_balloon1.txt --right_gt_path tools_gt/tools_right/P022_balloon1.txt --smoothing_method exp --save_images --infer_images --predict_tools --write_video
```
Running this bash command assumes a video is saved under videos/P022_balloon1.wmv, its main outputs are:
* Video - the labeled video will be located in `model_output/P022_bolloon1/labeled_video.mp4`
* Tool usage predictions files - will be located in `model_output/P022_bolloon1/tool_usage_prediction/left/predictions_exp.txt` (and the same way for the right hand)

The running will also produce the model's raw outputs for each frame (before smoothing) in model_output/P022_bolloon1/labels and the labeled video frames in `model_output/P022_bolloon1/labeled_images`.

## Detection Model Evaluation
In order to evaluate the trained model on the test set use:
```
python evaluate_model.py
```
This will create a dataframe consisting the AP@k and mAP@k for each class, for each k in {25,50,75} as requested, print it and save it.

## Tool Usage Evaluation
Given our tool usage predictions for each hand per frame and the corresponding ground-truth, we can run evaluate_tool_usage.py to get the desired evaluation metrics.
Example:
```bash
python evaluate_tool_usage.py --pred_file_path_left model_output/P022_balloon1/tool_usage_prediction/left/predictions_exp.txt --pred_file_path_right model_output/P022_balloon1/tool_usage_prediction/right/predictions_exp.txt --gt_file_path_left tools_gt/tools_left/P022_balloon1.txt --gt_file_path_right tools_gt/tools_right/P022_balloon1.txt --video_name P022_balloon1 --smoothing_method exp
```

This will produce an evaluating txt file including all metrics, for example:
<p align="center">
  <img src="https://github.com/TalIfargan/CVSA_HW1/blob/master/metrics_example.png" />
</p>

After producing all evaluation files, in our case 3 smoothing methods for any video (total of 15 files), we can run `mean_calculator.py` to get our final results, which are the mean metric over the videos for each desired metric.

