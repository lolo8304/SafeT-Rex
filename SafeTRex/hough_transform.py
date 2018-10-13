#from picamera.array import PiRGBArray
#from picamera import PiCamera

import time
import cv2
import numpy as np
import math
from lane_recognition.hough_transform_module import detect_lane
from lane_recognition.hough_transform_module import execute_pipeline_key
from lane_recognition.hough_configuration import configurations

from car_client import CarStateMachineClient
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
ap.add_argument("-i", "--imagesFolder", required=False, default=None,
	help="images directory and pattern to load")
ap.add_argument("-p", "--imagesPattern", required=False, default=None,
	help="images pattern of name to load")
ap.add_argument("-c", "--config", required=False, default="axahack2018",
	help="configuration pipeline name - default axahack2018")

args = vars(ap.parse_args())

class VideoInput():
    def __init__(self, videoName):
        self.__videoName = videoName
        self.__camera = cv2.VideoCapture(self.__videoName)
        time.sleep(0.1)
    def read(self):
        ret, image = self.__camera.read()
        return image



class ImagesInput():
    def __init__(self, imagesFolder, imagesPattern):
        self.__imagesFolder = imagesFolder
        self.__imagesPattern = imagesPattern
        print("images in ", imagesFolder, " look for ", imagesPattern)
        self.__images = [img for img in os.listdir(self.__imagesFolder) if img.startswith(imagesPattern)]
        self.__sorted = sorted(self.__images)
        print("images ", len(self.__sorted))
        self.__index = 0
        time.sleep(0.1)
    def read(self):
        if (self.__index < len(self.__sorted)):
            imageName = self.__sorted[self.__index]
            #print("read image ", os.path.join(self.__imagesFolder, imageName))
            image = cv2.imread(os.path.join(self.__imagesFolder, imageName))
            self.__index = self.__index + 1
            return image
        else:
            return None

inputMode = None
if (args["imagesFolder"]):
    inputMode = ImagesInput(args["imagesFolder"], args["imagesPattern"])
else:
    filename = "lane_recognition/data/test_videos/test-recording-1.h264"
    if (args["video"] is not None):
        filename = args["video"]
    inputMode = VideoInput(filename)

#camera = PiCamera()
#camera.resolution = (320, 240)
#camera.framerate = 16
#rawCapture = PiRGBArray(camera, size=(320, 240))
video = None

driver = CarStateMachineClient(recording=args["recording"], init=0, simulate=True)

def releaseVideo(video):
    video.release()

pipeline = args["config"]
image_config = configurations()[pipeline]


while(True):
    time.sleep(1 / 12) # 12 frames / s
    #image = frame.array
    image = inputMode.read()
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

        detect_lane(image, image_config, args["debug"], args["xdebug"], driver)

    key = cv2.waitKey(1) & 0xFF
    if (key != 255):
        pass
        #print("key=",key)
    if key == ord("q"):
        break
    found, image_config = execute_pipeline_key(key, image_config)

