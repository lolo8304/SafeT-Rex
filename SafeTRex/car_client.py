import requests
import time


class CarStateMachine():
    def __init__(self, url="http://192.168.6.107:5002", recordingNo=0, init=30, simulate=False):
        self.__state = ("RUN", init)
        self.__url = url
        self.__recordingNo = recordingNo
        self.__recordingFileName = ""
        if (self.__recordingNo > 0):
            self.__recordingFile = open("safet-rex-recording-motor-"+str(self.__recordingNo)+".csv", "w+")
        self.__simulate = simulate
        self.setRUN(init)
        self.lastSTOP = 0

    def SetRemoteValue(self, name, value):
        url = self.__url+'/'+name+"/"+str(value)
        if (self.__recordingNo > 0):
            print (format(time.time(), '.2f'), ", ", url, file=self.__recordingFile, flush=True)
        if not self.__simulate:
            response = requests.get(url)

    def SetRemoteCall(self, name):
        url = self.__url+'/'+name
        if (self.__recordingNo > 0):
            print (str(time.time()), ": ", url)
        if not self.__simulate:
            response = requests.get(url)

    def setRUN(self, tempo):
        if tempo == -1:
            tempo = self.__state[1]
        self.SetRemoteValue("speed", tempo)
        self.__state = ("RUN", tempo)

    def setSTOP(self):
        print("STOPSIGN!!!!")
        if time.time() - self.lastSTOP > 2:
            self.SetRemoteValue("speed", 0)
            self.__state = ("STOP", 0)
            print("stoping. . .")
            time.sleep(5)
            self.SetRemoteValue("speed", 30)
            self.__state = ("RUN", 30)
            self.lastSTOP = time.time()
        print("Last Stop to recent!")

    def setAngle(self, angle):
        self.SetRemoteValue("steer", -angle)

    def setREDLIGHT(self):
        print("Red Light")
        self.SetRemoteValue("speed", 0)
        self.__state = ("REDLIGHT", 0)

    def setGREENLIGHT(self):
        print("Go!")
        self.setRUN(30)

    def close(self):
        pass
