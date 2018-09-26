from ../pi-cam-cv/lane_recognition/hough_transform_module import detect_lane

def lanedetector(sr, driver):
    ld = LineDetector(sr, driver)
    ld.run()


class LineDetector():
    def __init__(self, sr, driver):
        self.__sr = sr
        self.__driver = driver

    def run(self):
        image = None
        while(image is None):
            image = self.__sr.currentimage
            detect_lane(image, self.__sr.isDebug())
