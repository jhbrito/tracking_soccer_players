#/usr/bin/env python3
# -- coding: utf-8 --

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import numpy as np
import os
import time
from os import path as osp

import tqdm
import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import instances


from torch.autograd import Variable
# Extract video properties
video = cv2.VideoCapture('C:/Users/EloiMartins/Desktop/resultados dos videos/detectro2/25sec_ataque.mp4')
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = video.get(cv2.CAP_PROP_FPS)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize video writer
video_writer = cv2.VideoWriter('C:/Users/EloiMartins/Desktop/resultados dos videos/detectro2/out.mp4', fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=float(frames_per_second), frameSize=(width, height), isColor=True)

# Initialize predictor
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

my_model=predictor.model














# Initialize visualizer
v = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), ColorMode.IMAGE)
print(v)

def runOnVideo(video, maxFrames):
    """ Runs the predictor on every frame in the video (unless maxFrames is given),
    and returns the frame with the predictions drawn.
    """

    readFrames = 0
    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            break

        # Get prediction results for this frame
        outputs = predictor(frame)
        f = open("C:/Users/EloiMartins/Desktop/det.txt", "a")

        category_3_detections = outputs["instances"].pred_classes ==0

        # Make sure the frame is colored
        indice = outputs["instances"].pred_classes
        indices=(indice==0).nonzero().squeeze() #all the positions on the array that are class 0 (person)

        persons=outputs["instances"].pred_boxes[indices]
        boxes_numpy=persons.tensor.cpu().numpy()
        boxes_numpy_round = np.around(boxes_numpy, decimals=1)

        scores_persons = outputs["instances"].scores[indices]
        scores_numpy=scores_persons.cpu().numpy()
        scores_numpy_round=np.around(scores_numpy,decimals=3)

        num_instancess=len(indices)



        cnt=1
        i=0



        y=0.999

        for s in range(0,num_instancess):

            l=boxes_numpy_round[i, 2]-boxes_numpy_round[i, 0]
            l_round=round(l,1)
            c=boxes_numpy_round[i, 3]-boxes_numpy_round[i, 1]
            c_round=round(c,1)
            print(readFrames,-1,boxes_numpy_round[i, 0],boxes_numpy_round[i, 1],l_round,c_round,scores_numpy_round[i],sep=',',file=f)

            i+=1










        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw a visualization of the predictions using the video visualizer
        visualization = v.draw_instance_predictions(frame, outputs["instances"].to("cpu"))

        # Convert Matplotlib RGB format to OpenCV BGR format
        visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)

        yield visualization

        readFrames += 1
        if readFrames > maxFrames:
            break

# Create a cut-off for debugging
#num_frames = 120


# Enumerate the frames of the video
for visualization in tqdm.tqdm(runOnVideo(video, num_frames), total=num_frames):

    # Write test image
    cv2.imwrite('POSE detectron2.png', visualization)

    # Write to video file
    video_writer.write(visualization)







# Release resources
video.release()
video_writer.release()
cv2.destroyAllWindows()