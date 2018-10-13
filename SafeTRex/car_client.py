#import import requests
from requests_futures.sessions import FuturesSession
import time

class CarStateMachineClient():
    def __init__(self, driver=None, url="http://localhost:5002", recording=0, init=30, simulate=False):
        self.__state = ("RUN", init)
        self.__driver = driver
        self.__url = url
        self.__recordingNo = recording
        self.__recordingFile = ""
        self.__recordingFileScript = ""
        self.__lastTime = time.time()
        self.__futureSession = FuturesSession()
        if (self.__recordingNo > 0):
            self.__recordingFile = open("safet-rex-recording-motor-"+str(self.__recordingNo)+".csv", "w+")
            self.__recordingFileScript = open("safet-rex-recording-script-"+str(self.__recordingNo)+".csv", "w+")
        self.__simulate = simulate
        if (init > -1):
            self.setRUN(init)
        self.lastSTOP = 0

    def SetRemoteValue(self, name, value):
        url = self.__url+'/'+name+"/"+str(value)
        if (self.__recordingNo > 0):
            t = time.time()
            print (format(t, '.2f'), ", ", name, ", ",value,", ", url, file=self.__recordingFile, flush=True)
            print ("echo sleep ", format((t - self.__lastTime), '.0f'), "; sleep ", format((t - self.__lastTime), '.0f'), file=self.__recordingFileScript, flush=True)
            print ("echo ",url, "; wget ", url, file=self.__recordingFileScript, flush=True)
            self.__lastTime = t
        if not self.__simulate:
            future_one = self.__futureSession.get(url)
            #requests.get(url)

    def SetRemoteCall(self, name):
        url = self.__url+'/'+name
        if (self.__recordingNo > 0):
            print (str(time.time()), ": ", url)
        if not self.__simulate:
            future_one = self.__futureSession.get(url)
            #requests.get(url)

    def setSpeed(self, speed):
        if speed == "faster" or speed == "inc":
            self.faster()
        elif speed == "slower" or speed == "dec":
            self.slower()
        else:
            self.setRUN(int(speed))

    def setRUN(self, tempo):
        if tempo == -1:
            tempo = self.__state[1]
        self.SetRemoteValue("speed", tempo)
        self.__state = ("RUN", tempo)

    def faster(self):
        if self.__driver: self.__driver.speed("faster")
        else: self.SetRemoteValue("speed", "faster")
    def slower(self):
        if self.__driver: self.__driver.speed("slower")
        else: self.SetRemoteValue("speed", "slower")


    def setSTOP(self):
        print("STOPSIGN!!!!")
        if time.time() - self.lastSTOP > 2:
            if self.__driver: self.__driver.speed(0)
            else: self.SetRemoteValue("speed", 0)
            self.__state = ("STOP", 0)
            print("stoping. . .")
            time.sleep(5)
            if self.__driver: self.__driver.speed(30)
            else: self.SetRemoteValue("speed", 30)
            self.__state = ("RUN", 30)
            self.lastSTOP = time.time()
        print("Last Stop to recent!")

    def setAngle(self, angle):
        if self.__driver: self.__driver.setAngle(-angle)
        else: self.SetRemoteValue("steer", -angle)
        
    def left(self):
        if self.__driver: self.__driver.left()
        else: self.SetRemoteValue("steer", "left")
    def right(self):
        if self.__driver: self.__driver.right()
        else: self.SetRemoteValue("steer", "right")

    def setREDLIGHT(self):
        print("Red Light")
        self.SetRemoteValue("speed", 0)
        self.__state = ("REDLIGHT", 0)

    def setGREENLIGHT(self):
        print("Go!")
        self.setRUN(30)

    def setSpeedFactor(self, value):
        self.SetRemoteValue("speedFactor", format(value, '.2f'))

    def incSpeedFactor(self):
        self.SetRemoteValue("speedFactor", "inc")
    def decSpeedFactor(self):
        self.SetRemoteValue("speedFactor", "dec")

    def close(self):
        pass
