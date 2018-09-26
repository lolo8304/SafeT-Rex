#from picamera.array import PiRGBArray
#import RPi.GPIO as GPIO
#from picamera import PiCamera
import time
import cv2
import numpy as np
import math
from Line import Line


# GPIO.setmode(GPIO.BOARD)
# GPIO.setup(7, GPIO.OUT)
# GPIO.setup(8, GPIO.OUT)
theta=0
#camera = cv2.VideoCapture("data/test_videos/lane1.h264")
camera = cv2.VideoCapture("data/test_videos/test2.h264")

#camera = PiCamera()
#camera.resolution = (320, 240)
#camera.framerate = 16
#rawCapture = PiRGBArray(camera, size=(320, 240))
time.sleep(0.1)

CARD_LONG_2_SHORT_FACTOR = 640 / 240
CARD_WRAP_LONG_MAX = 640
CARD_WRAP_SHORT_MAX = int(CARD_WRAP_LONG_MAX / CARD_LONG_2_SHORT_FACTOR)

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

def get_lane_lines(color_image, solid_lines=True):
    # grab the dimensions of the image and calculate the center
    # of the image
    (h, w) = image.shape[:2]
    h = int(h / 2)
    h3 = int(h / 2)
    crop_img = image[h3: h3 + h, 0:w]

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (17, 17), 0)
    edged = cv2.Canny(blurred, 85, 85)

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
                                rho=10,
                                theta=np.pi / 180,
                                threshold=2,
                                min_line_len=15,
                                max_line_gap=1)
    print(lines)

    if(lines is not None and lines.any() != None):
        # convert (x1, y1, x2, y2) tuples into Lines
        detected_lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3]) for l in lines]

        # if 'solid_lines' infer the two lane lines
        if solid_lines:
            candidate_lines = []
            for line in detected_lines:
                    # consider only lines with slope between 30 and 60 degrees
                    if 0.5 <= np.abs(line.slope) <= 2:
                        candidate_lines.append(line)
            # interpolate lines candidates to find both lanes
            lane_lines = compute_lane_from_candidates(candidate_lines, gray.shape)
        else:
            # if not solid_lines, just return the hough transform output
            lane_lines = detected_lines
        return crop_img, gray, blurred, edged, lane_lines
    else:
        return crop_img, gray, blurred, edged, []

#for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
while(True):

    #image = frame.array
    ret, image = camera.read()
    crop_img, gray, blurred, edged, lines = get_lane_lines(image, solid_lines=True)
   #lines = cv2.HoughLinesP(edged,1,np.pi/180,10,minLineLength,maxLineGap)

    #for x in range(0, len(lines)):
    print ("lines = ",lines)
    for line in lines:
            cv2.line(crop_img,(line.x1,line.y1),(line.x2,line.y2),(0,255,0),10)
            theta=theta+math.atan2((line.y2-line.y1),(line.x2-line.x1))
            #print("theta ", theta)
            # threshold = 6
            # if (theta>threshold):
            #     print ("turn left")
            # if(theta < -threshold):
            #     print("right")
            # if(abs(theta) < threshold):
            #     print ("straight")
            #     theta=0
            show_thumb("crop",crop_img, 0, 0)
            show_thumb("edge",edged, 2, 0)
            show_thumb("gray",gray, 4, 0)
            show_thumb("blurred",blurred, 0, 2)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break




#   print(theta)GPIO pins were connected to arduino for servo steering control
#    threshold=6
#    if(theta>threshold):
#        GPIO.output(7,True)
#        GPIO.output(8,False)
#        print("left")
#    if(theta<-threshold):
#        GPIO.output(8,True)
#        GPIO.output(7,False)
#        print("right")
#    if(abs(theta)<threshold):
#       GPIO.output(8,False)
#       GPIO.output(7,False)
#       print "straight"
#    theta=0
#    cv2.imshow("Frame",image)
#    key = cv2.waitKey(1) & 0xFF
#    rawCapture.truncate(0)
#    if key == ord("q"):
#        break
