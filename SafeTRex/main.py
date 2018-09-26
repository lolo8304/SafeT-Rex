from .car import *
from .objectdetection import *
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import threading

slow = 30
fast = 50
left = -30
right = 30


class CarHandler:
    def __init__(self):
        self.__driver = ServoCar()

    def start(self):
        self.__driver.speed(slow)
        sr = StreamReader()
        time.sleep(0.1)

        sign = threading.Thread(target=signdetection, args=[sr, self.__driver])
        lanes = None

        print("Starting SingDetection Thread...")
        sign.start()
        print("Starting StreamReader...")
        sr.run()



class StreamReader:
    def __init__(self):
        #self.__cam = cv2.VideoCapture(0)
        self.__cam = PiCamera()
        self.__cam.resolution = (320, 240)
        self.__cam.framerate = 16
        self.__rawCapture = PiRGBArray(self.__cam, size=(320, 240))
        time.sleep(1)

        #ret, self.currentimage = self.__cam.read()

    def run(self):
        for frame in self.__cam.capture_continuous(self.__rawCapture, format="bgr", use_video_port=True):
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
            self.currentimage = frame.array
        #while (True):
            #self.currentimage = self.__cam.read()


