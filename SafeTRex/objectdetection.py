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

    def __init__(self, sr):
        self.__sr = sr
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
                            (image.shape[1] - x_shift, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255),
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
        if debug:
            # draw a rectangle around the objects
            for (x_pos, y_pos, width, height) in cascade_obj:
                if self.__sr.isDebug():
                    cv2.rectangle(image, (x_pos + 5, y_pos + 5), (x_pos + width - 5, y_pos + height - 5),
                                  (255, 255, 255), 2)
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
                                cv2.putText(image, 'Red', (x_pos + 5, y_pos - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (0, 0, 255), 2)
                            self.red_light = True

                        # Green light
                        elif 5.5 / 8 * (height - 30) < maxLoc[1] < height - 30:
                            if self.__sr.isDebug():
                                cv2.putText(image, 'Green', (x_pos + 5, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (0, 255, 0),
                                            2)
                            self.green_light = True

                        # yellow light
                        # elif 4.0/8*(height-30) < maxLoc[1] < 5.5/8*(height-30):
                        #    cv2.putText(image, 'Yellow', (x_pos+5, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        #    self.yellow_light = True
        return v


class DetectLights():
    @staticmethod
    def get_green_circles(color_image):
        # grab the dimensions of the image and calculate the center
        # of the image
        (h, w) = color_image.shape[:2]
        # h = int(h / 2)
        # h3 = int(h / 2)
        # h3 = 0
        # crop_img = color_image[h3: h3 + h, 0:w]
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # blurred = cv2.GaussianBlur(gray, (17, 17), 0)
        # edged = cv2.Canny(blurred, 85, 85)
        lower_green = np.array([65, 60, 60])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        res = cv2.bitwise_and(color_image, color_image, mask=mask)
        ret, thrshed = cv2.threshold(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), 3, 255, cv2.THRESH_BINARY)
        # contours, hier = cv2.findContours(thrshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # for cnt in contours:
        #    cv2.putText(color_image, "Green Object Detected", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        #   cv2.rectangle(color_image, (5, 40), (400, 100), (0, 255, 255), 2)
        # if color_image.shape[-1] == 3:  # color image
        #    b, g, r = cv2.split(color_image)  # get b,g,r
        #    rgb_img = cv2.merge([r, g, b])  # switch it to rgb
        #    gray_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        # else:
        #    gray_img = color_image

        # img = cv2.medianBlur(gray_img, 5)
        # cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # with the arguments:
        # dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
        # lines: A vector that will store the parameters (x_{start}, y_{start}, x_{end}, y_{end}) of the detected lines
        # rho : The resolution of the parameter r in pixels. We use 1 pixel.
        # theta: The resolution of the parameter \theta in radians. We use 1 degree (CV_PI/180)
        # threshold: The minimum number of intersections to “detect” a line
        # maxLineGap: The maximum gap between two points to be considered in the same line.
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("IMG", gray)
        return cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 200, param1=70, param2=15, minRadius=0, maxRadius=20)
        # print(lines)

    @staticmethod
    def get_red_circles(color_image):
        # grab the dimensions of the image and calculate the center
        # of the image
        (h, w) = color_image.shape[:2]
        # h = int(h / 2)
        # h3 = int(h / 2)
        # h3 = 0
        # crop_img = color_image[h3: h3 + h, 0:w]
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # blurred = cv2.GaussianBlur(gray, (17, 17), 0)
        # edged = cv2.Canny(blurred, 85, 85)
        lower_red = np.array([0, 130, 130])
        upper_red = np.array([50, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        res = cv2.bitwise_and(color_image, color_image, mask=mask)
        ret, thrshed = cv2.threshold(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), 3, 255, cv2.THRESH_BINARY)
        # contours, hier = cv2.findContours(thrshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # for cnt in contours:
        #    cv2.putText(color_image, "Green Object Detected", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        #   cv2.rectangle(color_image, (5, 40), (400, 100), (0, 255, 255), 2)
        # if color_image.shape[-1] == 3:  # color image
        #    b, g, r = cv2.split(color_image)  # get b,g,r
        #    rgb_img = cv2.merge([r, g, b])  # switch it to rgb
        #    gray_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        # else:
        #    gray_img = color_image

        # img = cv2.medianBlur(gray_img, 5)
        # cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # with the arguments:
        # dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
        # lines: A vector that will store the parameters (x_{start}, y_{start}, x_{end}, y_{end}) of the detected lines
        # rho : The resolution of the parameter r in pixels. We use 1 pixel.
        # theta: The resolution of the parameter \theta in radians. We use 1 degree (CV_PI/180)
        # threshold: The minimum number of intersections to “detect” a line
        # maxLineGap: The maximum gap between two points to be considered in the same line.
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("IMG", gray)
        return cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 200, param1=80, param2=14, minRadius=0, maxRadius=20)
        # print(lines)


class StopHelper():
    def __init__(self):
        self.amount = 0
        self.lastdistance = -1

    def tester(self, distance):
        if (distance > self.lastdistance - 100 and distance < self.lastdistance + 100) or distance == -1:
            self.lastdistance = distance
            self.amount += 1
            if self.amount >= 5:
                self.amount = 0
                self.lastdistance = -1
                return True
            else:
                return False


class LightHelper():
    def __init__(self):
        self.amount = 0
        self.wasRed = False

    def tester(self, isRed):
        if isRed:
            self.amount += 1
        else:
            self.amount = 0
            if self.wasRed:
                self.wasRed = False
                return True
        if self.amount > 5:
            amount = 0
            self.wasRed = True
            return True
        else:
            return False



class SignDetector():
    def __init__(self, sr, driver):
        self.__sr = sr
        self.__driver = driver

    def run(self):
        self.obj_detection = ObjectDetection(self.__sr)
        # cascade classifiers
        self.stop_cascade = cv2.CascadeClassifier('SafeTRex/cascade_xml/stop_sign.xml')
        self.speed_cascade = cv2.CascadeClassifier('SafeTRex/cascade_xml/speed_sign.xml')

        # h1: stop sign
        self.h1 = 15.5 - 10  # cm
        # h2: traffic light
        self.h2 = 15.5 - 10
        # h3: speed sign
        self.h3 = 15.5 - 10

        self.d_to_camera = DistanceToCamera(self.__sr)
        self.d_stop_sign = 25
        self.d_light = 25
        self.d_speed = 25

        stophelper = StopHelper()
        lighthelper = LightHelper()

        while (True):
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
            image = self.__sr.getCurrentImage()
            grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # object detection
            v_param1 = self.obj_detection.detect(self.stop_cascade, grey_image, image, "STOP")
            v_param3 = self.obj_detection.detect(self.speed_cascade, grey_image, image, "Tempo 50")

            #Traffic Light detection
            circlesRed = DetectLights.get_red_circles(image)
            circlesGreen = DetectLights.get_green_circles(image)

            if self.__sr.isDebug():
                if circlesRed is not None:
                    for circle in circlesRed:
                        circles = [np.round(circle[0, :]).astype("int")]
                        # loop over the (x, y) coordinates and radius of the circles
                        for (x, y, r) in circles:
                            # draw the circle in the output image, then draw a rectangle
                            # corresponding to the center of the circle
                            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
                            cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

                if circlesGreen is not None and circlesGreen.any() != None:
                    for circle in circlesGreen:
                        circles = [np.round(circle[0, :]).astype("int")]
                        # loop over the (x, y) coordinates and radius of the circles
                        for (x, y, r) in circles:
                            # draw the circle in the output image, then draw a rectangle
                            # corresponding to the center of the circle
                            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
                            cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                    # show the output image

            if circlesRed != []:
                if lighthelper.tester(True):
                    self.__driver.setREDLIGHT()
            else:
                if lighthelper.tester(False):
                    self.__driver.setGREENLIGHT()
            d1 = 0
            d2 = 0
            d3 = 0
            d4 = 0

            # distance measurement
            if v_param1 > 0 or v_param3 > 0:
                if v_param1 > 0:
                    d1 = self.d_to_camera.calculate(v_param1, self.h1, 200, image)
                    if stophelper.tester(d1):
                        self.__driver.setSTOP()
                if v_param3 > 0:
                    d3 = self.d_to_camera.calculate(v_param3, self.h3, 200, image)
                    self.__driver.setRUN(50)
                self.d_stop_sign = d1
                self.d_light = d2
                self.d_speed = d3

                # if v_param1 > 0:
                print("v param 1 |STOPSIGN|=", v_param1, " distance=", d1)
                # if v_param3 > 0:
                print("v param 3 |TEMPOLIMIT|=", v_param3, " distance=", d3)

            if self.__sr.isDebug():
                cv2.imshow("ObjectDetection", image)
                key = cv2.waitKey(1) & 0xFF

            # clear the stream in preparation for the next frame

            # if the `q` key was pressed, break from the loop


