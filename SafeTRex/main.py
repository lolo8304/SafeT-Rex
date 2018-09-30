from .car_client import *
from .objectdetection import *
from .lanedetector import *
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import threading
import os
debug = True

slow = 30
fast = 50
left = -30
right = 30

class CarHandler:
    def __init__(self, args):
        self.__args = args
        self.__driver = CarStateMachine(recording=args["recording"], init=50)

    def start(self):
        self.__driver.setRUN(30)
        sr = StreamReader(self.__args)
        time.sleep(0.1)

        #sign = threading.Thread(target=signdetection, args=[sr, self.__driver])
        lanes = threading.Thread(target=lanedetector, args=[sr, self.__driver])

        #print("Starting SignDetection Thread...")
        #sign.start()
        print("Starting LaneDetector Thread...")
        lanes.start()
        print("Starting StreamReader...")
        sr.run()

def releaseVideo(video):
    video.release()


class StreamReader:
    def __init__(self, args):
        self.__debug = args["debug"]
        self.__recordingNo = args["recording"]
        self.__video = None

        #self.__cam = cv2.VideoCapture(0)
        self.__cam = PiCamera()
        dim = (320, 240)
        self.__cam.resolution = dim
        self.__cam.framerate = 16
        time.sleep(0.1)
        self.currentimage = None

        self.rawCapture = PiRGBArray(self.__cam, size=dim)


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

    def recordImage(self, image):
        if self.needsRecording():
            if self.__video is None:
                name = "safet-rex-recording-"+str(self.recordingNo())+".h264"
                print("start frame recording to ", name)
                (h, w) = image.shape[:2]
                if os.path.exists(name):
                    os.remove(name)
                self.__video = cv2.VideoWriter(name, cv2.VideoWriter_fourcc('h','2','6','4'), 10, (w,h))
                atexit.register(releaseVideo, self.__video)
            self.__video.write(image)


    def run(self):
        time.sleep(1)
        for frame in self.__cam.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
            self.currentimage = frame.array
            self.rawCapture.truncate(0)
            img = self.getCurrentImage()
            self.recordImage(img)
            if self.isDebug():
                # show the frame
                cv2.imshow("Frame", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        #while (True):
            #self.currentimage 
            # = self.__cam.read()


