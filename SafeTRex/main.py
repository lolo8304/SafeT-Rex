from .car_client import *
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
    def __init__(self, args):
        self.__args = args
        self.__driver = CarStateMachine(recording=args["recording"], init=30)

    def start(self):
        self.__driver.setRUN(30)
        sr = StreamReader(self.__args)
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
    def __init__(self, args):
        self.__debug = args["debug"]
        self.__recordingNo = args["recording"]

        #self.__cam = cv2.VideoCapture(0)
        self.__cam = PiCamera()
        self.__cam.resolution = (320, 240)
        self.__cam.framerate = 16
        time.sleep(0.1)
        self.currentimage = None

        self.rawCapture = PiRGBArray(self.__cam, size=(320, 240))


        #ret, self.currentimage = self.__cam.read()

    def getCurrentImage(self):
        image = None
        image = self.currentimage
        while(image is None):
            image = self.currentimage
        return image.copy()

    def isDebug(self):
        return self.__debug

    def recordingNo(self):
        return self.__recordingNo
    def needsRecording(self):
        return self.__recordingNo > 0

    def run(self):
        time.sleep(1)
        for frame in self.__cam.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
            self.currentimage = frame.array
            self.rawCapture.truncate(0)
            if self.isDebug():
                # show the frame
                cv2.imshow("Frame", self.getCurrentImage())
        #while (True):
            #self.currentimage 
            # = self.__cam.read()


