#================================================================
#
#   File name   : detection_custom.py
#   Author      : PyLessons
#   Created date: 2020-09-17
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : object detection image and video example
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from yolov3.configs import *

image_name = "image_10m/image_10m_2"
extension = ".jpg"
# video_path   = "./IMAGES/test.mp4"

i = 1

# while True:
#     try:
#         image_path   = f"./IMAGES/{image_name}/{image_name}_{i}"
#         output_path   = f"./IMAGES/{image_name}_results/{image_name}_{i}"
#         yolo = Load_Yolo_model()
#         detect_image(yolo, image_path + extension, output_path + "_detect" + extension, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
#         i += 1
#     except:
#         print("error")
#         break

image_path   = f"./IMAGES/{image_name}"
output_path   = f"./IMAGES/{image_name}"
yolo = Load_Yolo_model()
detect_image(yolo, image_path + extension, output_path + "_detect" + extension, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))

# detect_image(yolo, image_path + extension, image_path + "_detect" + extension, input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
#detect_video(yolo, video_path, './IMAGES/detected.mp4', input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
# detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))

# detect_video_realtime_mp(video_path, "Output.mp4", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0), realtime=False)
