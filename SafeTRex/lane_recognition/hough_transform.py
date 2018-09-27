#from picamera.array import PiRGBArray
#from picamera import PiCamera

import time
import cv2
import numpy as np
import math
from hough_transform_module import detect_lane

camera = cv2.VideoCapture("data/test_videos/test4.h264")

#camera = PiCamera()
#camera.resolution = (320, 240)
#camera.framerate = 16
#rawCapture = PiRGBArray(camera, size=(320, 240))
time.sleep(0.1)

#for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
while(True):
    time.sleep(0.01)
    #image = frame.array
    ret, image = camera.read()
    detect_lane(image, False)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

