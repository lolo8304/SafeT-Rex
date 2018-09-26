from .car import *
from .objectdetection import *
from .lanedetector import *
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import threading
debug = True

slow = 30
fast = 50
left = -30
right = 30

class CarHandler:
    def __init__(self):
        self.__driver = CarStateMachine()

    def start(self):
        self.__driver.setRUN(30)
        sr = StreamReader()
        time.sleep(0.1)

        sign = threading.Thread(target=signdetection, args=[sr, self.__driver])
        lanes = threading.Thread(target=lanedetector, args=[sr, self.__driver])

        print("Starting SignDetection Thread...")
        sign.start()
        print("Starting LaneDetector Thread...")
        lanes.start()
        print("Starting StreamReader...")
        sr.run()



class StreamReader:
    def __init__(self):
        #self.__cam = cv2.VideoCapture(0)
        self.__cam = PiCamera()
        self.__cam.resolution = (320, 240)
        self.__cam.framerate = 16
        time.sleep(0.1)
        self.currentimage = None

        self.rawCapture = PiRGBArray(self.__cam, size=(320, 240))


        #ret, self.currentimage = self.__cam.read()

    def run(self):
        time.sleep(1)
        for frame in self.__cam.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
            self.currentimage = frame.array
            self.rawCapture.truncate(0)
            if debug :
                # show the frame
                cv2.imshow("Frame", self.currentimage)
        #while (True):
            #self.currentimage 
            # = self.__cam.read()


