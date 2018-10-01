#from picamera.array import PiRGBArray
#import RPi.GPIO as GPIO
#from picamera import PiCamera
import time
import cv2
import numpy as np
from math import atan2
from math import atan
from math import degrees
from collections import deque
import os


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
        self.bias = self.compute_bias()

    def compute_slope(self):
        return (self.y2 - self.y1) / (self.x2 - self.x1 + np.finfo(float).eps)

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

    def draw(self, img, color=[0, 255, 255], thickness=5):

        # xx1 = max(0, self.x1)
        # yy1 = max(0, self.y1)
        # xx2 = max(0, self.x2)
        # yy2 = max(0, self.y2)
        # printD(xx1, yy1, xx2, yy2)
        # cv2.line(img, (xx1, yy1), (xx2, yy2), color, thickness)

        cv2.line(img, (self.x1, self.y1), (self.x2, self.y2), color, thickness)

    def degree(self):
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
        return self.degree() + line.degree()

        

time.sleep(0.1)

CARD_LONG_2_SHORT_FACTOR = 640 / 240
CARD_WRAP_LONG_MAX = 640
CARD_WRAP_SHORT_MAX = int(CARD_WRAP_LONG_MAX / CARD_LONG_2_SHORT_FACTOR)

debug = False
startTime = time.time()
isRaspi = os.uname()[4][:3] == "arm"

def printD(*objects):
    global startTime
    t = time.time() - startTime
    print(format(t, '.2f'), end="")
    print(objects)


def isDebug():
    global debug
    return debug

def setDebug(flag):
    global debug
    debug = flag

def show_thumb(name, image, x_index, y_index):
    """show tumbnail on screen to debug image pipeline"""

    MAX_WIDTH = CARD_WRAP_SHORT_MAX * 2
    # we need to keep in mind aspect ratio so the image does
    # not look skewed or distorted -- therefore, we calculate
    # the ratio of the new image to the old image
    r = MAX_WIDTH / image.shape[1]
    dim = (MAX_WIDTH, int(image.shape[0] * r))
    
    # perform the actual resizing of the image and show it
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow("Card Detector-"+name, resized);
    cv2.moveWindow("Card Detector-"+name, x_index * (dim[0] + 20), y_index * (dim[1] + 20));




def hough_lines_detection(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    return lines

def compute_lane_from_candidates(line_candidates, img_shape):
    """
    Compute lines that approximate the position of both road lanes.

    :param line_candidates: lines from hough transform
    :param img_shape: shape of image to which hough transform was applied
    :return: lines that approximate left and right lane position
    """

    # separate candidate lines according to their slope
    pos_lines = [l for l in line_candidates if l.slope > 0]
    neg_lines = [l for l in line_candidates if l.slope < 0]

    # interpolate biases and slopes to compute equation of line that approximates left lane
    # median is employed to filter outliers
    neg_bias = np.median([l.bias for l in neg_lines]).astype(int)
    neg_slope = np.median([l.slope for l in neg_lines])
    x1, y1 = 0, neg_bias
    x2, y2 = -np.int32(np.round(neg_bias / neg_slope)), 0
    left_lane = Line(x1, y1, x2, y2)

    # interpolate biases and slopes to compute equation of line that approximates right lane
    # median is employed to filter outliers
    lane_right_bias = np.median([l.bias for l in pos_lines]).astype(int)
    lane_right_slope = np.median([l.slope for l in pos_lines])
    x1, y1 = 0, lane_right_bias
    x2, y2 = np.int32(np.round((img_shape[0] - lane_right_bias) / lane_right_slope)), img_shape[0]
    right_lane = Line(x1, y1, x2, y2)

    return left_lane, right_lane

def isRationalNumber(f):
    return -1000.0 < f < 1000.0

def isRationalLine(line):
    return line and isRationalNumber(line.x1) and isRationalNumber(line.y1) and isRationalNumber(line.x2) and isRationalNumber(line.y2)

def get_lane_lines(color_image):
    global isRaspi
    # grab the dimensions of the image and calculate the center
    # of the image
    (h, w) = color_image.shape[:2]
    #h = int(h / 2)
    h3 = int(h / 3)
    #h3 = 0
    crop_img = color_image[0: h - h3, 0:w]
    gray = cv2.cvtColor(crop_img[0: h - h3, 0:w], cv2.COLOR_BGR2GRAY)
    blurred = None
    if False and isRaspi:
        gray_flip = cv2.flip(gray[0: h - h3, 0:w], flipCode=1)
        blurred = cv2.GaussianBlur(gray_flip[0: h - h3, 0:w], (17, 17), 0)
    else:
        blurred = cv2.GaussianBlur(gray[0: h - h3, 0:w], (11, 11), 0)
    edged = cv2.Canny(blurred[0: h - h3, 0:w], 65, 65)

# with the arguments:
# dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
# lines: A vector that will store the parameters (x_{start}, y_{start}, x_{end}, y_{end}) of the detected lines
# rho : The resolution of the parameter r in pixels. We use 1 pixel.
# theta: The resolution of the parameter \theta in radians. We use 1 degree (CV_PI/180)
# threshold: The minimum number of intersections to “detect” a line
# minLinLength: The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
# maxLineGap: The maximum gap between two points to be considered in the same line.
    lines = []
    lines = hough_lines_detection(img=edged,
                                rho=1,
                                theta=np.pi / 180,
                                threshold=1,
                                min_line_len=25,
                                max_line_gap=5)
    #printD(lines)

    if(lines is not None and lines.any() != None):
        # convert (x1, y1, x2, y2) tuples into Lines
        detected_lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3]) for l in lines]

        # if 'solid_lines' infer the two lane lines
        candidate_lines = []
        for line in detected_lines:
            # 0.5 = 63 degree, 1 = 45 degree, 2 = 26 degree, 3 = 18 degree
            # consider only lines with slope between 18 and 60 degrees
            if 0.5 <= np.abs(line.slope) <= 4:
                #printD("line candidate slope=", line.slope, " degree=", line.degree())
                line.draw(crop_img)
                candidate_lines.append(line)
        # interpolate lines candidates to find both lanes
        left, right = compute_lane_from_candidates(candidate_lines, gray.shape)
        return crop_img, gray, blurred, edged, left, right
    else:
        return crop_img, gray, blurred, edged, None, None


def line_intersection2(line1, line2):
    if (line1 is not None and line2 is not None):
        return line_intersection( ( (line1.x1, line1.y1), (line1.x2, line1.y2)), ((line2.x1, line2.y1), (line2.x2, line2.y2)) )
    else:
        return False, None


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

def drawLine(crop_img, line, color=(0,255,0), text=""):
    printD(text," (x,y)=", line.x1, line.y1, " (x,y)=", line.x2, line.y2)
    if isRationalLine(line):
        if isDebug():
            cv2.line(crop_img,(line.x1,line.y1),(line.x2,line.y2),color,10)

# from -1.0 via 0 to 1.0 (left to right)
def steering_directionX(intersection_point, left, right, image, defaultW = 0):
    x = intersection_point[0]
    h = 0
    w = defaultW
    if (image is not None):
        (h, w) = image.shape[:2]
    w2 = w / 2
    directionX = (x - w2) / w2
    if (directionX < 0):
        directionX = max(-1.0, directionX)
    else:
        directionX = min(1.0, directionX)
    return directionX


def draw_steering_angle(directionX, ticks, nofStraightTicks, crop_img):
    (h, w) = crop_img.shape[:2]
    ticksW = w / ticks // 2
    Line(0, 0, w // 2 - nofStraightTicks * ticksW, 0).draw(crop_img, color=[255, 0, 0])
    Line(w // 2 - nofStraightTicks * ticksW, 0, w // 2 + nofStraightTicks * ticksW, 0).draw(crop_img, color=[0, 0, 255])
    Line(w // 2 + nofStraightTicks * ticksW, 0, w, 0).draw(crop_img, color=[0, 255, 0])

# divide by 8 regions each side, first 2 regions each side - straight
# value between -45 - 45
def steering_angle(directionX, crop_img):
    ticks = 8
    straightTicks = 2
    draw_steering_angle(directionX, ticks, straightTicks, crop_img)
    maxAngle = 15
    directionX100 = int(directionX * maxAngle)
    absStraightDistance = maxAngle * straightTicks // ticks
    isStraight = abs(directionX100) <= absStraightDistance
    if isStraight:
        return "straight", directionX100
    elif directionX100 < 0:
        return "left", directionX100
    else:
        return "right", directionX100


def show_steering_angle(point, directionString, angle100, crop_img, offset = 0):
    #printD("smooth direction = ", directionString, " ", angle100, " crossed x=", point[0], " y=", point[1])
    (h, w) = crop_img.shape[:2]
    if isDebug():
        if directionString == "straight":
            cv2.putText(crop_img, directionString+","+str(angle100), (int(w/2)-30, 30+offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        elif directionString == "left":
            cv2.putText(crop_img, directionString+","+str(angle100), (30, 30+offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        elif directionString == "right":
            cv2.putText(crop_img, directionString+","+str(angle100), (w-140, 30+offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        elif directionString == "left-inc":
            cv2.putText(crop_img, directionString+","+str(angle100), (30, 30+offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else: #rihght-inc
            cv2.putText(crop_img, directionString+","+str(angle100), (w-160, 30+offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

def calculate_steering_angle(point, left, right, crop_img):
    directionX = steering_directionX( point, left, right, crop_img)
    directionString, angle100 = steering_angle(directionX, crop_img)
    printD("direction = ", directionString, " ", angle100, " crossed x=", point[0], " y=", point[1])
    return directionString, angle100

def drawArray(crop_img, point, directionX, color):
    cv2.circle(crop_img, point, 15, color, -1)
    dist = 30
    if directionX < 0:
        Line(point[0]-dist, point[1], point[0]+dist, point[1]).draw(crop_img, color)
        Line(point[0]-dist, point[1], point[0], point[1]+dist).draw(crop_img, color)
        Line(point[0]-dist, point[1], point[0], point[1]-dist).draw(crop_img, color)
    if directionX > 0:
        Line(point[0]-dist, point[1], point[0]+dist, point[1]).draw(crop_img, color)
        Line(point[0]+dist, point[1], point[0], point[1]-dist).draw(crop_img, color)
        Line(point[0]+dist, point[1], point[0], point[1]-dist).draw(crop_img, color)


## https://steemit.com/mathematics/@mes/video-notes-angle-between-two-lines-formula-in-terms-of-slopes
def calculate_steering_angle_from_single_line(point, left, right, crop_img):
    directionX = steering_directionX( point, left, right, crop_img)
    directionString, angle100 = steering_angle(directionX, crop_img)
    (h, w) = crop_img.shape[:2]

    left_degree = left.degree()
    right_degree = right.degree()
    deg = left.degree_between(right)

    printD("degree =", str(deg), " left=", left_degree, " right=", right_degree)
    inc = 3
    if left_degree == 0:
        #no line visible on left side, seems to be too far right -> go left
        drawArray(crop_img, ( int((left.x1 + point[0])// 2), int((left.y1 + point[1]) // 2) ), -1, (0,0,255))
        return "left-inc", inc
    elif right_degree == 0:
        #no line visible on right side, seems to be too far right -> go right
        p1 = ( right.fx(0), 0)
        drawArray(crop_img, ( int((right.x2 + p1[0])// 2), int((right.y2 - p1[1]) // 2) ), 1, (0,255,0))
        return "right-inc", inc
    elif (left_degree > right_degree + 10):
        drawArray(crop_img, ( int((left.x1 + point[0])// 2), int((left.y1 + point[1]) // 2) ), -1, (0,0,255))
        return "left-inc", inc
    elif (right_degree > left_degree + 10):
        p1 = ( right.fx(0), 0)
        drawArray(crop_img, ( int((right.x2 + p1[0])// 2), int((right.y2 - p1[1]) // 2) ), 1, (0,255,0))
        return "right-inc", inc
    else:
        cv2.circle(crop_img, point, 20, (0,0,255), -1)
        return "straight", 0


def test():
  testL = Line(0, 360, 240, 0)
  testR = Line(480, 360, 240, 0)
  crossed, point = line_intersection2(testL, testR)
  printD("test intersection ", point)

  testL = Line(0, 360, 240, 0)
  testR = Line(480, 360, 480, 0)
  crossed, point = line_intersection2(testL, testR)
  printD("test intersection ", point)

  p = (240, 0)
  testDirectionX = steering_directionX(p, testL, testR, None, 480)
  printD("test direct for ", p, " is ", testDirectionX)

  p = (-2147483600.0, -1994294800.0)
  testDirectionX = steering_directionX(p, testL, testR, None, 480)
  printD("test direct for ", p, " is ", testDirectionX)

  p = (480, 0)
  testDirectionX = steering_directionX(p, testL, testR, None, 480)
  printD("test direct for ", p, " is ", testDirectionX)

CONST_DIR = 0
CONST_ANGLE = 1
CONST_INC = 2
CONST_SMOOTH_DIR = 3
CONST_SMOOTH_ANGLE = 4

smooth_buffer = deque(maxlen=5)
last_element = None

def smooth_directionX(directionString, angle100):
    global last_element
    #printD("----------------------------------")
    new_element = [directionString, angle100, -1, directionString, angle100]
    if last_element is None:
        new_element[CONST_INC] = 1
        #printD("first element")
    else:
        if last_element[CONST_DIR] == directionString:
            new_element[CONST_INC] = last_element[CONST_INC] + 1
            #printD("SAME as before", new_element[CONST_INC])
            if last_element[CONST_INC] < 6:
                #printD("KEEP SMOOTH direction", last_element[CONST_INC])
                new_element[CONST_SMOOTH_DIR] = last_element[CONST_SMOOTH_DIR]
                new_element[CONST_SMOOTH_ANGLE] = last_element[CONST_SMOOTH_ANGLE]
            else:
                #printD("ADAPT SMOOTH DIRECTION", new_element[CONST_SMOOTH_DIR])
                new_element[CONST_SMOOTH_DIR] = new_element[CONST_DIR]
                new_element[CONST_SMOOTH_ANGLE] = new_element[CONST_ANGLE]
        else:
            #rint ("NEW direction - KEEP SMOOTH direction")
            new_element[CONST_INC] = 1
            new_element[CONST_SMOOTH_DIR] = last_element[CONST_SMOOTH_DIR]
            new_element[CONST_SMOOTH_ANGLE] = last_element[CONST_SMOOTH_ANGLE]
    last_element = new_element
    return new_element

inc = 0
lastMotorAngle = -142
lastMotorTime = -42.0

def allowedToSendToMotor(angle100):
    global lastMotorAngle
    global lastMotorTime
    t = time.time()
    tdiff = t - lastMotorTime
    if lastMotorAngle != angle100:
        printD("last=",lastMotorAngle, ", angle=", angle100, " ne=", lastMotorAngle != angle100)
        if tdiff > 1:
            lastMotorAngle = angle100
            lastMotorTime = t
            return True
    return False

def sendToMotor(angle100, driver = None):
    if allowedToSendToMotor(angle100):
        printD("send to motor ", angle100)
        if driver is not None:
            driver.setAngle(angle100)

def allowedToSendIncrementToMotor():
    global lastMotorTime
    t = time.time()
    tdiff = t - lastMotorTime
    if tdiff > 1:
        lastMotorTime = t
        return True
    return False


def sendIncrementToMotor(directionX, angle100, driver = None):
    if allowedToSendIncrementToMotor():
        if directionX == "left-inc":
            printD("send left to motor ", angle100)
            if driver is not None:
                driver.left()
        elif directionX == "right-inc":
            printD("send right to motor ", angle100)
            if driver is not None:
                driver.right()
        else:
            printD("send to motor ", angle100)
            if driver is not None:
                driver.setAngle(angle100)


def detect_lane(image, debugFlag = False, driver = None):
    global inc 
    setDebug(debugFlag)
    #printD("------------")
    crop_img, gray, blurred, edged, left, right = get_lane_lines(image)
    (h, w) = crop_img.shape[:2]

    directionString = None
    angle100 = 0
    point = (0, 0)
    crossed = False
    if (isRationalLine(left) and isRationalLine(right)):
        crossed, point = line_intersection2(left, right)
        if (crossed):
            drawLine(crop_img, left, (0,0,255), "left")
            drawLine(crop_img, right, (0,255,0), "right")
            directionString, angle100 = calculate_steering_angle_from_single_line(point, left, right, crop_img)
    elif isRationalLine(left):
        #virtual_horizon = Line(w, 0, w, h)
        virtual_horizon = Line(0, 0, w, 0)
        crossed, point = line_intersection2(left, virtual_horizon)
        if crossed:
            drawLine(crop_img, left, (0,0,255), "left")
            virtual_horizon = Line(point[0], 0, point[0], h)
            drawLine(crop_img, virtual_horizon, (255,0,0), "virtual")
            directionString, angle100 = calculate_steering_angle_from_single_line(point, left, virtual_horizon, crop_img)
    elif isRationalLine(right):
        #virtual_horizon = Line(0, 0, 0, h)
        virtual_horizon = Line(0, 0, w, 0)
        crossed, point = line_intersection2(virtual_horizon, right)
        if crossed:
            virtual_horizon = Line(point[0], h, point[0], 0)
            drawLine(crop_img, virtual_horizon, (255,0,0), "virtual")
            drawLine(crop_img, right, (0,255,0), "right")
            directionString, angle100 = calculate_steering_angle_from_single_line(point, virtual_horizon, right, crop_img)

    if not crossed:
        #printD("no crossed lines")
        return

    #show_steering_angle(point, directionString, angle100, crop_img, 50)

    new_element = smooth_directionX(directionString, angle100)
    show_steering_angle(point, new_element[CONST_SMOOTH_DIR], new_element[CONST_SMOOTH_ANGLE], crop_img)
    sendIncrementToMotor(new_element[CONST_SMOOTH_DIR], new_element[CONST_SMOOTH_ANGLE], driver)

    #time.sleep(0.02)
    if isDebug():
        show_thumb("crop",crop_img, 0, 0)
    #show_thumb("edge",edged, 1, 0)
    #show_thumb("gray",gray, 0, 1)
    #show_thumb("gray",gray, 0, 1)
    #show_thumb("blurred",blurred, 1, 1)

