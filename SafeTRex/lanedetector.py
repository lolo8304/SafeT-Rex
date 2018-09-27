from .lane_recognition.hough_transform_module import detect_lane
import time

def lanedetector(sr, driver):
    ld = LineDetector(sr, driver)
    ld.run()


class LineDetector():
    def __init__(self, sr, driver):
        self.__sr = sr
        self.__driver = driver
        print("starting lanedetector")

    def run(self):
        start_time = time.time() # start time of the loop
        while(True):
            image = None
            while(image is None):
                image = self.__sr.currentimage
            detect_lane(image, self.__sr.isDebug())
            print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop