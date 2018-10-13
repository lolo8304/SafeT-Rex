from .lane_recognition.hough_transform_module import detect_lane
from .lane_recognition.hough_transform_module import execute_pipeline_key
from .lane_recognition.hough_configuration import configurations

import time

def lanedetector(sr, driver):
    ld = LineDetector(sr, driver)
    ld.run()


class LineDetector():
    def __init__(self, sr, driver):
        self.__sr = sr
        self.__driver = driver
        print("starting lanedetector")
        self.__image_config = configurations()["axahack2018"]


    def run(self):
        while(not self.__sr.stopEvent.isSet()):
            image  = self.__sr.getCurrentImage()
            detect_lane(image, self.__image_config, self.__sr.isDebug(), self.__sr.isXDebug(), self.__driver)
            time.sleep(0.5)
