__author__ = 'jkminder'

from .main import *
# import the necessary packages
# from picamera.array import PiRGBArray
# from picamera import PiCamera
import time
import math
import numpy as np
import cv2
debug = True
detect_threshold = 8


# from .features import red_green_yellow
# Neighbor Count

def signdetection(sr, driver):
    sd = SignDetector(sr, driver)
    sd.run()


class DistanceToCamera(object):

    def __init__(self):
        # camera params
        self.alpha = 8.0 * math.pi / 180
        self.v0 = 119.865631204
        self.ay = 332.262498472

    def calculate(self, v, h, x_shift, image):
        # compute and return the distance from the target point to the camera
        d = h / math.tan(self.alpha + math.atan((v - self.v0) / self.ay))
        if d > 0:
            if self.__sr.isDebug():
                cv2.putText(image, "%.1fcm" % d,
                        (image.shape[1] - x_shift, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                        2)
        return d


class ObjectDetection(object):

    def __init__(self, sr):
        self.__sr = sr
        self.red_light = False
        self.green_light = False
        self.yellow_light = False

    def detect(self, cascade_classifier, gray_image, image, text="obj", force_rect=False,
               default_threshold=detect_threshold):

        # y camera coordinate of the target point 'P'
        v = 0

        # minimum value to proceed traffic light state validation
        threshold = 0

        # detection
        cascade_obj = cascade_classifier.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=default_threshold,
            minSize=(7, 7),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if debug :
        # draw a rectangle around the objects
            for (x_pos, y_pos, width, height) in cascade_obj:
                if self.__sr.isDebug():
                    cv2.rectangle(image, (x_pos + 5, y_pos + 5), (x_pos + width - 5, y_pos + height - 5), (255, 255, 255), 2)
                v = y_pos + height - 5
                # print(x_pos+5, y_pos+5, x_pos+width-5, y_pos+height-5, width, height)

                # stop sign
                if 0.85 < width / height < 1.15 or force_rect:
                    if self.__sr.isDebug():
                        cv2.putText(image, text, (x_pos, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # traffic lights
                else:
                    roi = gray_image[y_pos + 10:y_pos + height - 10, x_pos + 10:x_pos + width - 10]
                    mask = cv2.GaussianBlur(roi, (25, 25), 0)
                    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)

                    # check if light is on
                    if maxVal - minVal > threshold:
                        if self.__sr.isDebug():
                            cv2.circle(roi, maxLoc, 5, (255, 0, 0), 2)

                        # Red light
                        if 1.0 / 8 * (height - 30) < maxLoc[1] < 4.0 / 8 * (height - 30):
                            if self.__sr.isDebug():
                                cv2.putText(image, 'Red', (x_pos + 5, y_pos - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            self.red_light = True

                        # Green light
                        elif 5.5 / 8 * (height - 30) < maxLoc[1] < height - 30:
                            if self.__sr.isDebug():
                                cv2.putText(image, 'Green', (x_pos + 5, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                        2)
                            self.green_light = True

                        # yellow light
                        # elif 4.0/8*(height-30) < maxLoc[1] < 5.5/8*(height-30):
                        #    cv2.putText(image, 'Yellow', (x_pos+5, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        #    self.yellow_light = True
        return v


class SignDetector():
    def __init__(self, sr, driver):
        self.__sr = sr
        self.__driver = driver

    def run(self):
        self.obj_detection = ObjectDetection(self.__sr)
        # cascade classifiers
        self.stop_cascade = cv2.CascadeClassifier('SafeTRex/cascade_xml/stop_sign.xml')
        self.light_cascade = cv2.CascadeClassifier('SafeTRex/cascade_xml/traffic_light.xml')
        self.speed_cascade = cv2.CascadeClassifier('SafeTRex/cascade_xml/speed_sign.xml')


        # h1: stop sign
        self.h1 = 15.5 - 10  # cm
        # h2: traffic light
        self.h2 = 15.5 - 10
        # h3: speed sign
        self.h3 = 15.5 - 10


        self.d_to_camera = DistanceToCamera()
        self.d_stop_sign = 25
        self.d_light = 25
        self.d_speed = 25

        while (True):
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
            image = self.__sr.getCurrentImage()
            grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # object detection
            v_param1 = self.obj_detection.detect(self.stop_cascade, grey_image, image, "STOP")
            v_param2 = self.obj_detection.detect(self.light_cascade, grey_image, image)
            v_param3 = self.obj_detection.detect(self.speed_cascade, grey_image, image, "Tempo 50")
            v_param4 = self.obj_detection.detect(self.light_cascade, grey_image, image, "Light Signal", True,
                                                 default_threshold=8)
            d1 = 0
            d2 = 0
            d3 = 0
            d4 = 0
            # distance measurement
            if v_param1 > 0 or v_param2 > 0 or v_param3 > 0 or v_param4 > 0:
                if v_param1 > 0:
                    d1 = self.d_to_camera.calculate(v_param1, self.h1, 200, image)
                    self.__driver.setSTOP()
                if v_param2 > 0:
                    d2 = self.d_to_camera.calculate(v_param2, self.h2, 100, image)
                if v_param3 > 0:
                    d3 = self.d_to_camera.calculate(v_param3, self.h3, 200, image)
                    self.__driver.setRUN(50)
                if v_param4 > 0:
                    d4 = self.d_to_camera.calculate(v_param4, self.h3, 100, image)
                self.d_stop_sign = d1
                self.d_light = d2
                self.d_speed = d3

            #if v_param1 > 0:
                print("v param 1 |STOPSIGN|=", v_param1, " distance=", d1)
            #if v_param2 > 0:
                print("v param 2 |LIGHTSIGNAL|=", v_param2, " distance=", d2)
            #if v_param3 > 0:
                print("v param 3 |TEMPOLIMIT|=", v_param3, " distance=", d3)



            key = cv2.waitKey(1) & 0xFF

            # clear the stream in preparation for the next frame


            # if the `q` key was pressed, break from the loop
