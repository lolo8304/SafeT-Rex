from flask import Flask
from flask_restful import Resource, Api
from flask import request
from .car import *
import sys
import threading

app = Flask(__name__)
api = Api(app)
__driver = CarStateMachine(0)
GPIO.setwarnings(False)

eventStopper = None

def shutdown_server():
    global eventStopper
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    eventStopper.shutdown()
    sys.exit(1)
    exit()


def getDriver():
    global __driver
    return __driver

class Hello(Resource):
    def get(self):
        return {'hello back': "hello"}  

class Stop(Resource):
    def get(self):
        shutdown_server()
        return "{'killed':'true'}"


class Speed(Resource):
    def get(self, speed):
        if speed == "faster" or speed == "inc":
            getDriver().faster()
        elif speed == "slower" or speed == "dec":
            getDriver().slower()
        else:
            getDriver().setRUN(int(speed))
        return {'speed': getDriver().getSpeed()}  
        
class SpeedFactor(Resource):
    def get(self, speedFactor):
        if speedFactor == "inc":
            getDriver().incSpeedFactor()
        elif speedFactor == "dec":
            getDriver().decSpeedFactor()
        else:
            getDriver().setSpeedFactor(float(speedFactor))
        return {'speedFactor': getDriver().getSpeedFactor()}  


class Steer(Resource):
    def get(self, steer):
        if steer == "left" or steer == "dec":
            getDriver().left()
        elif steer == "right" or steer == "inc":
            getDriver().right()
        else:
            getDriver().setAngle(int(steer))
        return {'steer': getDriver().getAngle()}  


api.add_resource(Hello, '/hello')  
api.add_resource(Stop, '/kill')  
api.add_resource(Speed, '/speed/<speed>')  
api.add_resource(SpeedFactor, '/speedFactor/<speedFactor>')  
api.add_resource(Steer, '/steer/<steer>')

def startServer(sr=None):
    global eventStopper
    eventStopper = sr
    print("server is starting on port 5002")
    app.run(host= '0.0.0.0', port='5002')

