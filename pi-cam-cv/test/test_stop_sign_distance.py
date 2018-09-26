__author__ = 'lolo8304'

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import math
import cv2


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
            cv2.putText(image, "%.1fcm" % d,
                (image.shape[1] - x_shift, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return d


class ObjectDetection(object):

    def __init__(self):
        self.red_light = False
        self.green_light = False
        self.yellow_light = False

    def detect(self, cascade_classifier, gray_image, image):

        # y camera coordinate of the target point 'P'
        v = 0

        # minimum value to proceed traffic light state validation
        threshold = 150    
        
        # detection
        cascade_obj = cascade_classifier.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # draw a rectangle around the objects
        for (x_pos, y_pos, width, height) in cascade_obj:
            cv2.rectangle(image, (x_pos+5, y_pos+5), (x_pos+width-5, y_pos+height-5), (255, 255, 255), 2)
            v = y_pos + height - 5
            #print(x_pos+5, y_pos+5, x_pos+width-5, y_pos+height-5, width, height)

            # stop sign
            if 0.85 < width/height < 1.15:
                cv2.putText(image, 'STOP', (x_pos, y_pos-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # traffic lights
            else:
                roi = gray_image[y_pos+10:y_pos + height-10, x_pos+10:x_pos + width-10]
                mask = cv2.GaussianBlur(roi, (25, 25), 0)
                (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)
                
                # check if light is on
                if maxVal - minVal > threshold:
                    cv2.circle(roi, maxLoc, 5, (255, 0, 0), 2)
                    
                    # Red light
                    if 1.0/8*(height-30) < maxLoc[1] < 4.0/8*(height-30):
                        cv2.putText(image, 'Red', (x_pos+5, y_pos-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        self.red_light = True
                    
                    # Green light
                    elif 5.5/8*(height-30) < maxLoc[1] < height-30:
                        cv2.putText(image, 'Green', (x_pos+5, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        self.green_light = True
    
                    # yellow light
                    #elif 4.0/8*(height-30) < maxLoc[1] < 5.5/8*(height-30):
                    #    cv2.putText(image, 'Yellow', (x_pos+5, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    #    self.yellow_light = True
        return v


class DetectionHandler():

    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    #camera.resolution = (640, 480)
    #camera.framerate = 32
    #rawCapture = PiRGBArray(camera, size=(640, 480))

    camera.resolution = (240, 180)
    camera.framerate = 16
    rawCapture = PiRGBArray(camera, size=(240, 180))

    # allow the camera to warmup
    time.sleep(0.1)

    obj_detection = ObjectDetection()


    # h1: stop sign
    h1 = 15.5 - 10  # cm
    # h2: traffic light
    h2 = 15.5 - 10

    # cascade classifiers
    stop_cascade = cv2.CascadeClassifier('cascade_xml/stop_sign.xml')
    light_cascade = cv2.CascadeClassifier('cascade_xml/traffic_light.xml')
    
    d_to_camera = DistanceToCamera()
    d_stop_sign = 25
    d_light = 25

    def handle(self):

      # capture frames from the camera
      for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # object detection
        v_param1 = self.obj_detection.detect(self.stop_cascade, grey_image, image)
        v_param2 = self.obj_detection.detect(self.light_cascade, grey_image, image)
        d1 = 0
        d2 = 0
        # distance measurement
        if v_param1 > 0 or v_param2 > 0:
          if v_param1 > 0:
            d1 = self.d_to_camera.calculate(v_param1, self.h1, 200, image)
          if v_param2 > 0:
            d2 = self.d_to_camera.calculate(v_param2, self.h2, 100, image)
          self.d_stop_sign = d1
          self.d_light = d2

        print ("v param 1=",v_param1, " distance=",d1)
        print ("v param 2=",v_param2, " distance=",d2)

        # show the frame
        cv2.imshow("Frame", image)

        key = cv2.waitKey(1) & 0xFF
      
        # clear the stream in preparation for the next frame
        self.rawCapture.truncate(0)
      
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
          break


DetectionHandler().handle()
