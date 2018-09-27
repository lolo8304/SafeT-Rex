import RPi.GPIO as GPIO
import time
import sys


class Servo(object):
    """Ansteuerung von Servos per PWM"""

    def __init__(self, gpio, hz=50, start=90, min=0, max=180):
        if min > max:
            tmp = min
            min = max
            max = tmp
        if min < 0:
            min = 0
        if max > 180:
            max = 180

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(gpio, GPIO.OUT)
        GPIO.setwarnings(False)
        self.io = GPIO.PWM(gpio, hz)  # 50 Hz
        self.io.start(start)
        self.min = min
        self.max = max

    def hello(self, name):
        print('Servo ' + name + ' from ' + str(self.min) + ' to ' + str(self.max))

    def angle(self, angle):
        #if angle < self.min:
        #    angle = self.min
        #elif angle > self.max:
        #    angle = self.max
        self.io.ChangeDutyCycle(float(angle) / 18 + 2.5)
        #time.sleep(0.2)
        print('Servo angle ' + str(angle))

    def pulse(self, angle):
        #if angle < self.min:
        #   angle = self.min
        #elif angle > self.max:
        #    angle = self.max
        self.io.ChangeDutyCycle(float(angle))
        #time.sleep(0.2)
        print('Servo pulse ' + str(angle))

    def close(self):
        self.io.stop()
        GPIO.cleanup()


class ServoCar(object):
    """Ansteuerung eines Modellautos mit Steuer und Speed"""

    def __init__(self):
        # Steuerung -100 ... 100
        self.__steer = Servo(23, 50, 7.5, 45, 135)
        self.__steerFactor = float(self.__steer.max) / 100

        # Geschwindigkeit 0...100
        self.__speed = Servo(24, 10, 0, 0, 100)
        self.__speedFactor = 0.25

    def hello(self):
        print('CAR: Hello from car')
        self.__steer.hello('Lenkung')
        self.__speed.hello('Geschwindigkeit')

    def steer(self, value):
        print("CAR: steer to " + str(value))
        self.__steer.angle(90 + self.__steerFactor * float(value))

    def speed(self, value):
        print("CAR: Set Speed to " + str(value))
        self.__speed.pulse(self.__speedFactor * float(value))


class CarStateMachine():
    def __init__(self):
        self.__state = ("RUN", 30)
        self._car = ServoCar()
        self.setRUN(30)
        self.lastSTOP = 0

    def setRUN(self, tempo):
        if tempo == -1:
            tempo = self.__state[1]
        self._car.speed(tempo)
        self.__state = ("RUN", tempo)

    def setSTOP(self):
        print("STOPSIGN!!!!")
        if time.time() - self.lastSTOP > 30:
            self._car.speed(0)
            self.__state = ("STOP", 0)
            print("stoping. . .")
            time.sleep(5)
            self._car.speed(30)
            self.__state = ("RUN", 30)
            self.lastSTOP = time.time()
        print("Last Stop to recent!")

    def setAngle(self, angle):
        self._car.steer(angle)

    def setREDLIGHT(self):
        print("Red Light")
        self._car.speed(0)
        self.__state = ("REDLIGHT", 0)

    def setGREENLIGHT(self):
        print("Go!")
        self.setRUN(30)


