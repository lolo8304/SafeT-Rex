#import RPi.GPIO as GPIO
import time
import sys, getopt


class Servo(object):
    """Ansteuerung von Servos per PWM"""
    def __init__(self, gpio, min=0, max=180):
        if min>max:
            tmp=min
            min=max
            max=tmp
        if min<0: min=0
        if max>180: max=180
	    #GPIO.setmode(GPIO.BCM)
	    #GPIO.setup(gpio, GPIO.OUT)
        #self.io = GPIO.PWM(gpio, 50) # 50 Hz
        #self.io.start(7.5)
        self.min=min
        self.max=max

    def hello(self, name):
        print('Servo '+name+' from '+str(self.min)+' to '+str(self.max))


    def go(self,angle):
        if angle<self.min: angle=self.min
        elif angle>self.max: angle=self.max
        #self.io.ChangeDutyCycle(float(angle)/18+2.5)
        time.sleep(0.2)
        print('Servo angle '+str(angle))

    def close(self):
        pass
        #self.io.stop()
        #GPIO.cleanup()


class ServoCar(object):
    """Ansteuerung eines Modellautos mit Steuer und Speed"""
    def __init__(self):
        self.__steer = Servo(23, 45, 135)
        self.__steerFactor = float(self.__steer.max)/100
        self.__speed = Servo(24)
        self.__speedFactor = float(self.__speed.max)/255/4.5

    def hello(self):
        print('Hello from car')
        self.__steer.hello('Lenkung')
        self.__speed.hello('Geschwindigkeit')

    def steer(self, value):
        self.__steer.go(90+self.__steerFactor*float(value))

    def speed(self, value):
        self.__speed.go(90+self.__speedFactor*float(value))




