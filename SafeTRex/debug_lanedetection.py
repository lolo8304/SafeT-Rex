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
from picamera.array import PiRGBArray
from picamera import PiCamera

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--debug", required=False, type=bool, default=False,
	help="debug mode - will show debug windows")
ap.add_argument("-x", "--xdebug", required=False, type=bool, default=False,
	help="X debug mode - will show detailed information about lane detection")
ap.add_argument("-r", "--recording", required=False, type=int, default=0,
	help="number to write files accordinly - suffix to filename")
ap.add_argument("-v", "--video", required=False, default=None,
	help="input video file name for running")
ap.add_argument("-i", "--imagesFolder", required=False, default=None,
	help="input images directory and pattern to load")
ap.add_argument("-p", "--imagesPattern", required=False, default=None,
	help="images pattern of name to load e.g. central*")
ap.add_argument("-c", "--config", required=False, default="axahack2018",
	help="configuration pipeline name - default axahack2018 - see hough_configuration.py")

args = vars(ap.parse_args())

class VideoInput():
    def __init__(self, videoName):
        self.__videoName = videoName
        self.__camera = cv2.VideoCapture(self.__videoName)
        time.sleep(0.1)
    def read(self):
        ret, image = self.__camera.read()
        return image


class PiCamInput():
    def __init__(self):
        self.__videoName = "picam"
        self.__camera = PiCamera()
        self.__camera.resolution = (320, 240)
        self.__camera.framerate = 24

        # allow the camera to warmup
        time.sleep(1.0)

    def read(self):
        image = np.empty((240 * 320 * 3,), dtype=np.uint8)
        self.__camera.capture(image, 'bgr')
        image2 = cv2.flip(image, flipCode=1)
        cv2.imshow("Frame", image2)
        print("get image from pi")
        return image2



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
    if (args["video"] is not None):
        filename = args["video"]
        if filename == "picam":
            inputMode = PiCamInput()
        else:
            inputMode = VideoInput(filename)
    else:
        filename = "lane_recognition/data/test_videos/demo-training-01.h264"
        inputMode = VideoInput(filename)

# global variable to record video to be able to release at end of program
video = None
# driver controls the car, here we need only simulation, no real calls to car happening
driver = CarStateMachineClient(recording=args["recording"], init=0, simulate=True)

# on exit handler at end of recording while stopping python program
def releaseVideo(video):
    video.release()

# initialize our pipeline configuration in file ./lane_recognition/hough_configuration.py
pipeline = args["config"]
image_config = configurations()[pipeline]

while(True):
    # recorded files from videos are written with 12 frames per second on pi
    time.sleep(1 / 12) # 12 frames / s
    image = inputMode.read()
    if (image is not None):

        # if recording of video is switched on
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

    # check if key pressed fits to a pipeline image configuration 
    # and will directly execute the needed command
    found, image_config = execute_pipeline_key(key, image_config)

