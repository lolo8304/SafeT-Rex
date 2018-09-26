#import RPi.GPIO as GPIO
import time
import sys, getopt


class Servo(object):
    """Ansteuerung von Servos per PWM"""

    def __init__(self, gpio, min=0, max=180):
        if min > max:
            tmp = min
            min = max
            max = tmp
        if min < 0:
            min = 0
        if max > 180:
            max = 180


        #GPIO.setmode(GPIO.BCM)
        #PIO.setup(gpio, GPIO.OUT)
        #self.io = GPIO.PWM(gpio, 50)  # 50 Hz
        #self.io.start(7.5)
        self.min = min
        self.max = max

    def hello(self, name):
        print('Servo ' + name + ' from ' + str(self.min) + ' to ' + str(self.max))

    def go(self, angle):
        if angle < self.min:
            angle = self.min
        elif angle > self.max:
            angle = self.max
        #self.io.ChangeDutyCycle(float(angle) / 18 + 2.5)
        time.sleep(0.2)
        print('Servo angle ' + str(angle))

    def close(self):
        pass
        #self.io.stop()
        #GPIO.cleanup()


class ServoCar(object):
    """Ansteuerung eines Modellautos mit Steuer und Speed"""

    def __init__(self):
        self.__steer = Servo(23, 45, 135)
        self.__steerFactor = float(self.__steer.max) / 100
        self.__speed = Servo(24)
        self.__speedFactor = float(self.__speed.max) / 255 / 4.5

    def hello(self):
        print('Hello from car')
        self.__steer.hello('Lenkung')
        self.__speed.hello('Geschwindigkeit')

    def steer(self, value):
        self.__steer.go(90 + self.__steerFactor * float(value))

    def speed(self, value):
        print("Set Speed to" + str(value))
        self.__speed.go(90 + self.__speedFactor * float(value))


class CarStateMachine():
    def __init__(self, car):
        self.__state = ("RUN", 30)
        self.setRUN(30)
        self._car = ServoCar()

    def setState(self, state, tempo=-1):
        # RUN
        if state == "RUN":
            self.setRUN(tempo)

    def setRUN(self, tempo):
        if tempo == -1:
            tempo = self.__state[1]
        self._car.speed(tempo)
        self.__state = ("RUN", tempo)

    def setSTOP(self):
        self._car.speed(0)
        self.__state = ("STOP", 0)
