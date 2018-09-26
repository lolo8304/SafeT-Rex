from .car import *
from .objectdetection import signdetection
import cv2
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
        self.__cam = cv2.VideoCapture(0)
        ret, self.currentimage = self.__cam.read()

    def run(self):
        while (True):
            self.currentimage = self.__cam.read()


