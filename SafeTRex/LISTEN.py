from flask import Flask
from flask_restful import Resource, Api
from car import *

app = Flask(__name__)
api = Api(app)
__driver = CarStateMachine(0)
GPIO.setwarnings(False)

def getDriver():
    global __driver
    return __driver

class Hello(Resource):
    def get(self):
        return {'hello back': "hello"}  


class Speed(Resource):
    def get(self, speed):
        getDriver().setRUN(int(speed))
        return {'speed': int(speed)}  
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
        if steer == "left":
            getDriver().left()
        elif steer == "right":
            getDriver().right()
        else:
            getDriver().setAngle(int(steer))
        return {'steer': int(steer)}  


api.add_resource(Hello, '/hello')  
api.add_resource(Speed, '/speed/<speed>')  
api.add_resource(SpeedFactor, '/speedFactor/<speedFactor>')  
api.add_resource(Steer, '/steer/<steer>')

if __name__ == '__main__':
    app.run(host= '0.0.0.0', port='5002')
