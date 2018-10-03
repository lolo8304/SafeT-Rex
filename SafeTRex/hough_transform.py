#from picamera.array import PiRGBArray
#from picamera import PiCamera

import time
import cv2
import numpy as np
import math
from lane_recognition.hough_transform_module import detect_lane
from car_client import CarStateMachine
import atexit
import os

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--debug", required=False, type=bool, default=False,
	help="debug mode")
ap.add_argument("-x", "--xdebug", required=False, type=bool, default=False,
	help="X debug mode")
ap.add_argument("-r", "--recording", required=False, type=int, default=0,
	help="number to write files accordinly")
ap.add_argument("-v", "--video", required=False, default=None,
	help="video file name for running")
args = vars(ap.parse_args())

filename = "lane_recognition/data/test_videos/test-recording-1.h264"
if (args["video"] is not None):
    filename = args["video"]
camera = cv2.VideoCapture(filename)

#camera = PiCamera()
#camera.resolution = (320, 240)
#camera.framerate = 16
#rawCapture = PiRGBArray(camera, size=(320, 240))
time.sleep(0.1)
video = None

driver = CarStateMachine(recording=args["recording"], init=0, simulate=True)

def releaseVideo(video):
    video.release()


#for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

while(True):
    time.sleep(0.0625) # 16 frames / s
    #image = frame.array
    ret, image = camera.read()
    if (image is not None):

        if args["recording"] > 0:
            if video is None:
                name = "safet-rex-recording-"+str(args["recording"])+".h264"
                print("start frame recording to ", name)
                (h, w) = image.shape[:2]
                if os.path.exists(name):
                    os.remove(name)
                video = cv2.VideoWriter(name, cv2.VideoWriter_fourcc('h','2','6','4'), 10, (w,h))
                atexit.register(releaseVideo, video)
            video.write(image)

        detect_lane(image, args["debug"], args["xdebug"], driver)

    key = cv2.waitKey(0) & 0xFF
    if (key != 255):
        pass
        #print("key=",key)
    if key == ord("q"):
        break

