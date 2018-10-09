#from picamera.array import PiRGBArray
#import RPi.GPIO as GPIO
#from picamera import PiCamera
import time
import cv2
import numpy as np
from math import atan2
from math import atan
from math import degrees
from math import sqrt
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

    def printXD(self, text):
        printXD(text," (x,y)=", self.x1, self.y1, " (x,y)=", self.x2, self.y2)

        

time.sleep(0.1)

CARD_LONG_2_SHORT_FACTOR = 640 / 240
CARD_WRAP_LONG_MAX = 640
CARD_WRAP_SHORT_MAX = int(CARD_WRAP_LONG_MAX / CARD_LONG_2_SHORT_FACTOR)

debug = False
xdebug = False
startTime = time.time()
isRaspi = os.uname()[4][:3] == "arm"

def printD(*objects):
    global startTime
    t = time.time() - startTime
    print(format(t, '.2f'), ": ", end="")
    print(objects, flush=True)

def printXD(*objects):
    if (isXDebug()):
        printD(objects)


def isDebug():
    global debug
    return debug
def isXDebug():
    global xdebug
    return xdebug

def setDebug(dFlag, xdFlag):
    global debug, xdebug
    debug = dFlag
    xdebug = xdFlag

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


#### ------------------------------------------
# pipelines START
#### ------------------------------------------


## crop image
##         "algorithm" : "bottom", "top"
##         "factor" : 0.33333333
def pipeline_Crop(image, parameters):
    alg = parameters["algorithm"]
    factor = parameters["factor"]
    (h, w) = image.shape[:2]
    image = image.copy()
    if alg == "bottom":
        image = image[0:h - int(h * factor), 0:w]
    elif alg == "top":
        image = image[int(h * factor):h, 0:w]
    return image

# no parameters
def pipeline_Grey(image, parameters):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# no parameters
##         "channel" : 0 = H, 1 = L, 2 = S
def pipeline_HLS(image, parameters):
    channel = parameters["channel"]
    channels_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    channels = cv2.split(channels_image)
    return channels[channel]

# no parameters
##         "channel" : 0 = L, 1 = U, 2 = V
def pipeline_LUV(image, parameters):
    channel = parameters["channel"]
    channels_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    channels = cv2.split(channels_image)
    return channels[channel]

# no parameters
##         "channel" : 0 = L, 1 = A, 2 = B
def pipeline_LAB(image, parameters):
    channel = parameters["channel"]
    channels_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L,A,B = cv2.split(channels_image)
    return L


# based on https://www.linkedin.com/pulse/advanced-lane-finding-pipeline-tiba-razmi/
def pipeline_HLS_LUV_LAB(image, parameters):
    channel1 = parameters["channel1"]
    channel2 = parameters["channel2"]
    channel3 = parameters["channel3"]
    channels_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    HLS_channels = cv2.split(channels_image)
    channels_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    LUV_channels = cv2.split(channels_image)
    channels_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    LAB_channels = cv2.split(channels_image)
    result_image = cv2.merge((HLS_channels[channel1], LUV_channels[channel2], LAB_channels[channel3]))
    return result_image



#          "size" : 17,
#          "sigma" : 0
def pipeline_GaussianBlur(image, parameters):
    size = parameters["size"]
    sigma = parameters["sigma"]
    return cv2.GaussianBlur(image, (size, size), sigma)

#          "size" : 17
def pipeline_medianBlur(image, parameters):
    size = parameters["size"]
    return cv2.medianBlur(image, size)

#          "kernel" : 3
def pipeline_SelectiveGaussianBlur(image, parameters):
    kernel = parameters["kernel"]
    return cv2.bilateralFilter(image, kernel, kernel * 2, kernel / 2)


#          "max" : 3
#          "method" : 0 oder 1
#          "size" : 5
def pipeline_AdaptiveThreshold(image, parameters):
    max = parameters["max"]
    method = parameters["method"]
    size = parameters["size"]
    image = image.copy()
    cv2.adaptiveThreshold(image, max, method, cv2.THRESH_BINARY, size, 0.0)
    return image


#          "threshold" : 3
#          "max" : 0 oder 1
#          "type" : 0
def pipeline_Threshold(image, parameters):
    threshold = parameters["threshold"]
    max = parameters["max"]
    type = parameters["type"]
    image = image.copy()
    ret, image2 = cv2.threshold(image, threshold, max, type)
    return image2


#          "threshold1" : 65.0,
#          "threshold2" : 65.0,
#          "apertureSize" : 3,
#          "L2gradient" : False
def pipeline_Canny(image, parameters):
    threshold1 = parameters["threshold1"]
    threshold2 = parameters["threshold2"]
    apertureSize = parameters["apertureSize"]
    L2gradient = parameters["L2gradient"]
    canny = cv2.Canny(image, threshold1, threshold2, apertureSize=apertureSize, L2gradient=L2gradient )
    #cv2.imshow("canny_121",canny)
    return canny


# warp_left: int
# warp_right: int
def pipeline_Warp(image, parameters):
    (IMAGE_H, IMAGE_W) = image.shape[:2]

    #IMAGE_H = 192
    #IMAGE_W = 640

    #IMAGE_WARP_LEFT = 207
    #IMAGE_WARP_RIGHT = 405

    IMAGE_WARP_LEFT = parameters["warp_left"]
    IMAGE_WARP_RIGHT = parameters["warp_right"]

    src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    dst = np.float32([[IMAGE_WARP_LEFT, IMAGE_H], [IMAGE_WARP_RIGHT, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

    #img = img[450:(450+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop
    warped_img = cv2.warpPerspective(image, M, (IMAGE_W, IMAGE_H)) # Image warping    
    return warped_img


def pipeline_process(image, image_config):
    #printD("pipeline configuration:",image_config["title"])
    configs = image_config["pipeline"]

    pipeline_results = []
    pipeline_result = {
        "title" : image_config["title"],
        "image" : image,
        "images" : pipeline_results
    }
    pipeline_results.append( {
        "type" : "origin",
        "title" : "origin",
        "parameters" : [],
        "image" : image
    })

    draw = False
    if configs and isinstance(configs, list):
        idx = 0
        for config in configs:
            type = config["type"]
            params = config["parameters"]
            off = "off" in config and config["off"]
            if not off:
                #printD("pipeline: ",type, " params: ", params)
                image = globals()['pipeline_'+type](image, params)
                if "draw" in config and config["draw"]:
                    draw = True
                    pipeline_result["draw_image"] = image
                pipeline_results.append( {
                    "type" : type,
                    "title" : type + "-"+str(idx),
                    "parameters" : params,
                    "image" : image
                })
                idx = idx + 1
    if not draw:
        pipeline_result["draw"] = pipeline_result["image"]
    pipeline_result["result_image"] = image
    return pipeline_result

def pipeline_show_thumb(pipeline_result):
    images = pipeline_result["images"]
    idx = 0
    for image in images:
        show_thumb(image["title"],image["image"], idx % 2, idx // 2)
        idx = idx + 1


def execute_pipeline_key(key_in, image_config):
    configs = image_config["pipeline"]
    if configs and isinstance(configs, list):
        for config in configs:
            off = "off" in config and config["off"]
            if not off and "keys" in config:
                for key, object in config["keys"].items():
                    if ord(key) == key_in:
                        config["parameters"][object["name"]] = object["f"](config["parameters"])
                        printD("execute ",config["type"]," key=", key, " new ",object["name"],"=", config["parameters"][object["name"]])
                        return True, image_config
    return False, image_config

#### ------------------------------------------
# pipelines END
#### ------------------------------------------


def hough_lines_detection(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    return lines

### start closest lines


def find_longest_none_zero(array, size):
    # example [0 [obj] 0 0 0 0 0 0 [obj] [obj] [obj,obj] [obj,obj,obj] 0 [obj] [obj,obj] ]
    cc = np.zeros([size], int)
    i = 0
    maxC = 0
    maxI = -1
    minI = 0
    minMaxI = 0
    for a in array:
      if i > 0 and len(a) > 0:
          if cc[i-1] == 0:
            minI = i
          cc[i] = cc[i-1] + len(a)
      else:
          cc[i] = len(a)
      if cc[i] > maxC:
        maxC = cc[i]
        maxI = i
        minMaxI = minI
      i = i+1
    return minMaxI, maxI, maxC

def keep_longest_non_zero(array, size):
  minIndex, maxIndex, max = find_longest_none_zero(array, size)
  cc = []
  for i in range(minIndex, maxIndex+1):
    cc.extend(array[i])
  return cc

def count_distance(array, f, max, distance):
    max_index = max // distance + 1
    count = [[] for _ in range(max_index)]
    for a in array:
        i = int( f(a) // distance)
        count[i].append(a)
    return count

def keep_closests(array, f, max, distance):
    cc = count_distance(array, f, max, distance)
    return keep_longest_non_zero(cc, len(cc))

### end closest lines

def compute_lane_from_candidates(line_candidates, crop_img, img_shape):
    """
    Compute lines that approximate the position of both road lanes.

    :param line_candidates: lines from hough transform
    :param img_shape: shape of image to which hough transform was applied
    :return: lines that approximate left and right lane position
    """

    # separate candidate lines according to their slope
    pos_lines = [l for l in line_candidates if l.slope > 0]
    pos_lines_closests = keep_closests(pos_lines, lambda line: line.x1, img_shape[1], 30 )
    pos_lines = pos_lines_closests
    for pos in pos_lines_closests:
        pass
        #pos.draw(crop_img, thickness=5)

    neg_lines = [l for l in line_candidates if l.slope < 0]
    neg_lines_closests = keep_closests(neg_lines, lambda line: line.x1, img_shape[1], 30 )
    neg_lines = neg_lines_closests
    for neg in neg_lines_closests:
        pass
        #neg.draw(crop_img, thickness=5)

    # interpolate biases and slopes to compute equation of line that approximates left lane
    # median is employed to filter outliers
    neg_bias = np.average([l.bias for l in neg_lines]).astype(int)
    neg_slope = np.average([l.slope for l in neg_lines])
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

def get_lane_lines(color_image, image_config):
    global isRaspi
    pipeline_result = pipeline_process(color_image, image_config)
    crop_img = pipeline_result["draw_image"]
    result_image = pipeline_result["result_image"]
    (h, w) = result_image.shape[:2]

# with the arguments:
# dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
# lines: A vector that will store the parameters (x_{start}, y_{start}, x_{end}, y_{end}) of the detected lines
# rho : The resolution of the parameter r in pixels. We use 1 pixel.
# theta: The resolution of the parameter \theta in radians. We use 1 degree (CV_PI/180)
# threshold: The minimum number of intersections to “detect” a line
# minLinLength: The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
# maxLineGap: The maximum gap between two points to be considered in the same line.
    lines = []
    lines = hough_lines_detection(img=result_image,
                                rho=1,
                                theta=np.pi / 180,
                                threshold=1,
                                min_line_len=20,
                                max_line_gap=10)

    if(lines is not None and lines.any() != None):
        # convert (x1, y1, x2, y2) tuples into Lines
        detected_lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3]) for l in lines]

        # if 'solid_lines' infer the two lane lines
        candidate_lines = []
        for line in detected_lines:
            # 0.5 = 63 degree, 1 = 45 degree, 2 = 26 degree, 3 = 18 degree
            # consider only lines with slope between 18 and 60 degrees
            if 0.2 <= np.abs(line.slope) <= 4:
                #printD("line candidate slope=", line.slope, " degree=", line.degree())
                line.draw(crop_img, color=(0, 255, 0), thickness=2)
                candidate_lines.append(line)
            else:
                #print("missing slope=",np.abs(line.slope))
                line.draw(crop_img, color=(255, 255, 255), thickness=2)
        # interpolate lines candidates to find both lanes
        left, right = compute_lane_from_candidates(candidate_lines, crop_img, crop_img.shape)
        return pipeline_result, left, right
    else:
        return pipeline_result, None, None


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

def drawLine(crop_img, line, color=(0,255,0), text="", tickness=4):
    line.printXD(text)
    if isRationalLine(line):
        if isDebug():
            cv2.line(crop_img,(line.x1,line.y1),(line.x2,line.y2),color,tickness)


inc = 0
lastMotorAngle = -142
lastMotorTime = -42.0
lastLeftLine = None
lastRightLine = None


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
    #draw_steering_angle(directionX, ticks, straightTicks, crop_img)
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
    printXD("direction = ", directionString, " ", angle100, " crossed x=", point[0], " y=", point[1])
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
    global lastLeftLine
    global lastRightLine
    directionX = steering_directionX( point, left, right, crop_img)
    directionString, angle100 = steering_angle(directionX, crop_img)
    (h, w) = crop_img.shape[:2]

    left_degree = left.degree()
    right_degree = right.degree()
    deg = left.degree_between(right)

    #printXD("degree =", str(deg), " diff=", (abs(left_degree - right_degree))," left=", left_degree, " right=", right_degree)
    inc = 3
    if left_degree == 0:
        lastRightLine = right
        #no line visible on left side, seems to be too far right -> go left
        #drawArray(crop_img, ( int((left.x1 + point[0])// 2), int((left.y1 + point[1]) // 2) ), -1, (0,0,255))
        return "straight", 0
    elif right_degree == 0:
        lastLeftLine = left
        #no line visible on right side, seems to be too far right -> go right
        p1 = ( right.fx(0), 0)
        #drawArray(crop_img, ( int((right.x2 + p1[0])// 2), int((right.y2 - p1[1]) // 2) ), 1, (0,255,0))
        return "straight", 0
    elif (15 < right_degree - left_degree):
        lastLeftLine = left
        lastRightLine = right
        #drawArray(crop_img, ( int((left.x1 + point[0])// 2), int((left.y1 + point[1]) // 2) ), -1, (0,0,255))
        return "right-inc", inc
    elif (15 < left_degree - right_degree):
        lastLeftLine = left
        lastRightLine = right
        p1 = ( right.fx(0), 0)
        #drawArray(crop_img, ( int((right.x2 + p1[0])// 2), int((right.y2 - p1[1]) // 2) ), 1, (0,255,0))
        return "left-inc", inc
    else:
        lastLeftLine = left
        lastRightLine = right
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
    printXD("----------------------------------")
    new_element = [directionString, angle100, -1, directionString, angle100]
    if last_element is None:
        new_element[CONST_INC] = 1
        #printD("first element")
    else:
        if last_element[CONST_DIR] == directionString:
            new_element[CONST_INC] = last_element[CONST_INC] + 1
            #printD("SAME as before", new_element[CONST_INC])
            if last_element[CONST_INC] < 3:
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


def allowedToSendToMotor(angle100):
    global lastMotorAngle
    global lastMotorTime
    t = time.time()
    tdiff = t - lastMotorTime
    if lastMotorAngle != angle100:
        printD("last=",lastMotorAngle, ", angle=", angle100, " ne=", lastMotorAngle != angle100)
        if tdiff > 0.5:
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

def calculate_steering_angle_from_double_line(crop_img, left, right):
    global lastLeftLine
    global lastRightLine
    crossed, point = line_intersection2(left, right)
    lastLeftLine = left
    lastRightLine = right
    if (crossed):
        drawLine(crop_img, left, (0,0,255), "left", 4)
        drawLine(crop_img, right, (255,0,0), "right", 4)
        directionString, angle100 = calculate_steering_angle_from_single_line(point, left, right, crop_img)
        return crossed, directionString, angle100
    return crossed, None, None

def detect_lane(image, image_config, debugFlag = False, xdebugFlag = False, driver = None):
    global inc 
    
    setDebug(debugFlag, xdebugFlag)
    #printD("------------")
    pipeline_result, left, right = get_lane_lines(image, image_config)
    crop_img = pipeline_result["draw_image"]
    (h, w) = crop_img.shape[:2]

    directionString = None
    angle100 = 0
    point = (0, 0)
    crossed = False
    if (isRationalLine(left) and isRationalLine(right)):
        crossed, directionString, angle100 = calculate_steering_angle_from_double_line(crop_img, left, right)
    elif isRationalLine(left):
        #printXD("only left")
        if True or lastRightLine is None:
            #virtual_horizon = Line(w, 0, w, h)
            virtual_horizon = Line(0, 0, w, 0)
            crossed, point = line_intersection2(left, virtual_horizon)
            if crossed:
                virtual_horizon = Line(point[0], 0, point[0], h)
                drawLine(crop_img, left, (0,0,255), "left")
                drawLine(crop_img, virtual_horizon, (255,0,0), "virtual")
                directionString, angle100 = calculate_steering_angle_from_single_line(point, left, virtual_horizon, crop_img)
        else:
            crossed, directionString, angle100 = calculate_steering_angle_from_double_line(crop_img, left, lastRightLine)

    elif isRationalLine(right):
        #printXD("only right")
        if True or lastLeftLine is None:
            #virtual_horizon = Line(0, 0, 0, h)
            virtual_horizon = Line(0, 0, w, 0)
            crossed, point = line_intersection2(virtual_horizon, right)
            if crossed:
                virtual_horizon = Line(point[0], h, point[0], 0)
                drawLine(crop_img, virtual_horizon, (255,0,0), "virtual")
                drawLine(crop_img, right, (0,255,0), "right")
                directionString, angle100 = calculate_steering_angle_from_single_line(point, virtual_horizon, right, crop_img)
        else:
            crossed, directionString, angle100 = calculate_steering_angle_from_double_line(crop_img, lastLeftLine, right)


    if not crossed:
        #printD("no crossed lines")
        return

    #show_steering_angle(point, directionString, angle100, crop_img, 50)

    new_element = smooth_directionX(directionString, angle100)
    #show_steering_angle(point, new_element[CONST_SMOOTH_DIR], new_element[CONST_SMOOTH_ANGLE], crop_img)
    sendIncrementToMotor(new_element[CONST_SMOOTH_DIR], new_element[CONST_SMOOTH_ANGLE], driver)

    time.sleep(0.02)
    if isDebug():
        pipeline_show_thumb(pipeline_result)

