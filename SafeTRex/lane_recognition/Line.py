import numpy as np
import cv2
import math
from math import atan2
from math import atan
from math import degrees
from math import sqrt


class Line:
    """
    A Line is defined from two points (x1, y1) and (x2, y2) as follows:
    y - y1 = (y2 - y1) / (x2 - x1) * (x - x1)
    Each line has its own slope and intercept (bias).
    """
    def __init__(self, x1, y1, x2, y2):

        self.x1 = np.float32(x1)
        self.y1 = np.float32(y1)
        self.x2 = np.float32(x2)
        self.y2 = np.float32(y2)

        self.slope = self.compute_slope()
        self.degree = self.compute_degree()
        self.bias = self.compute_bias()
        self.length = self.compute_length()

    def compute_slope(self):
        return (self.y2 - self.y1) / (self.x2 - self.x1 + np.finfo(float).eps)

    def compute_length(self):
        return sqrt((self.y2 - self.y1) ** 2 + (self.x2 - self.x1) ** 2)

    def compute_bias(self):
        return self.y1 - self.slope * self.x1

    def get_coords(self):
        return np.array([self.x1, self.y1, self.x2, self.y2])

    def set_coords(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    # y = mx+b
    def fy(self, x):
        return self.slope * x + self.bias

    # x = (y-b) / m
    def fx(self, y):
        return (y - self.bias) / self.slope

    def draw(self, img, color=(0, 255, 255), thickness=3):

        # xx1 = max(0, self.x1)
        # yy1 = max(0, self.y1)
        # xx2 = max(0, self.x2)
        # yy2 = max(0, self.y2)
        # printD(xx1, yy1, xx2, yy2)
        # cv2.line(img, (xx1, yy1), (xx2, yy2), color, thickness)

        cv2.line(img, (self.x1, self.y1), (self.x2, self.y2), color, thickness)

    def compute_degree(self):
        if self.slope < 1.0e+10:
            # - because of missorientation of y1,y2 - origin - left, top
            tangent_angle = degrees(atan(self.slope))
            if self.y2 < self.y1:
                tangent_angle = -tangent_angle
            return 90 - tangent_angle
        else:
            return 0

    # see https://www.youtube.com/watch?v=O8M4ZErxE-M
    def degree_between(self, line): 
        return self.degree + line.degree

    def lamdaF(self):
        if self.bias > 0:
            return "y = x * "+str(self.slope)+" + "+str(self.bias)
        elif self.bias < 0:
            return "y = x * "+str(self.slope)+" - "+str(-self.bias)
        else:
            return "y = x * "+str(self.slope)

    def printXD(self, text):
        printXD(text," (x,y)=", self.x1, self.y1, " (x,y)=", self.x2, self.y2)

    def draw_filled_area(self, img, line, color=(127,255,0), thickness=3):
        (h, w) = img.shape[:2]
        overlay = img.copy()
        #calc x crossed point at position y = h
        xh = self.fx(h)
        points = [
            (xh, h),
            (self.x2, self.y2),
            (line.x1, line.y1),
            (line.x2, line.y2),
        ]
        cv2.fillPoly(overlay, np.int_([points]), color)
        opacity = 0.4
        cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
