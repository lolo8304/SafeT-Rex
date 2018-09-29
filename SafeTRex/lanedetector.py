from .lane_recognition.hough_transform_module import detect_lane
import random
import atexit
import os
import cv2

def lanedetector(sr, driver):
    ld = LineDetector(sr, driver)
    ld.run()


def releaseVideo(video):
    video.release()

class LineDetector():
    def __init__(self, sr, driver):
        self.__sr = sr
        self.__driver = driver
        self.__video = None
        print("starting lanedetector")

    def run(self):
        while(True):
            image  = self.__sr.getCurrentImage()
            if self.__sr.needsRecording():
                if self.__video is None:
                    name = "safet-rex-recording-"+str(self.__sr.recordingNo())+".h264"
                    print("start frame recording to ", name)
                    (h, w) = image.shape[:2]
                    if os.path.exists(name):
                        os.remove(name)
                    video = cv2.VideoWriter(name, cv2.VideoWriter_fourcc('h','2','6','4'), 10, (w,h))
                    atexit.register(releaseVideo, video)
                video.write(image)
            detect_lane(image, self.__sr.isDebug(), self.__driver)
