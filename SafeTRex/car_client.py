import requests


def SetRemoteValue(name, value):
    url = 'http://192.168.6.107:5002/'+name+"/"+str(value)
    response = requests.get(url)

def SetRemoteCall(name):
    url = 'http://192.168.6.107:5002/'+name
    response = requests.get(url)


class CarStateMachine():
    def __init__(self, init=30):
        self.__state = ("RUN", init)
        self.setRUN(init)
        self.lastSTOP = 0

    def setRUN(self, tempo):
        if tempo == -1:
            tempo = self.__state[1]
        SetRemoteValue("speed", tempo)
        self.__state = ("RUN", tempo)

    def setSTOP(self):
        print("STOPSIGN!!!!")
        if time.time() - self.lastSTOP > 2:
            SetRemoteValue("speed", 0)
            self.__state = ("STOP", 0)
            print("stoping. . .")
            time.sleep(5)
            SetRemoteValue("speed", 30)
            self.__state = ("RUN", 30)
            self.lastSTOP = time.time()
        print("Last Stop to recent!")

    def setAngle(self, angle):
        SetRemoteValue("steer", -angle)

    def setREDLIGHT(self):
        print("Red Light")
        SetRemoteValue("speed", 0)
        self.__state = ("REDLIGHT", 0)

    def setGREENLIGHT(self):
        print("Go!")
        self.setRUN(30)

    def close(self):
        pass
