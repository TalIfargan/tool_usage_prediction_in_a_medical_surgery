import cv2
import numpy as np
import bbox_visualizer as bbv

## https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/      ## basic opencv tutorial
## https://github.com/shoumikchow/bbox-visualizer  ## bbox_visualizer git with examples

cap = cv2.VideoCapture("P022_balloon1.wmv")

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
i=0
while (cap.isOpened()):
    i += 1
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # add bounding boxes
        # bbox = [xmin, ymin, xmax, ymax]
        boxes1 = [[50, 150, 200, 300],[80, 200, 250, 450]]
        boxes2 = [[190, 260, 330+2*i, 440+i]]

        frame = bbv.draw_multiple_rectangles(frame, boxes1,bbox_color=(255,0,0))
        frame = bbv.add_multiple_labels(frame, ["label1","label1"], boxes1,text_bg_color=(255,0,0))
        frame = bbv.draw_multiple_rectangles(frame, boxes2,bbox_color=(0,250,0))
        frame = bbv.add_multiple_labels(frame, ["label2"], boxes2,text_bg_color=(0,250,0))

        # Display the resulting frame
        # cv2.imshow('Frame', frame)

        # Press Q on keyboard to  exit
        # if cv2.waitKey(33) & 0xFF == ord('q'):
        #     break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
# cv2.destroyAllWindows()