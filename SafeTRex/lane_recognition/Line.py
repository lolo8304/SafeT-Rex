import numpy as np
import cv2
import math
from math import atan2
from math import atan
from math import degrees
from math import sqrt

#Finds the intersection of two lines, or returns false.
#The lines are defined by (o1, p1) and (o2, p2).
def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    div = det(xdiff, ydiff)
    if div == 0:
       return False, None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return True, (x, y)

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
        print(text," (x,y)=", self.x1, self.y1, " (x,y)=", self.x2, self.y2)

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


    #Finds the intersection of two lines, or returns false.
    #The lines are defined by (o1, p1) and (o2, p2).
    def line_intersection(self, line2):
        if (line2 is not None):
            return line_intersection( ( (self.x1, self.y1), (self.x2, self.y2)), ((line2.x1, line2.y1), (line2.x2, line2.y2)) )
        else:
            return False, None



