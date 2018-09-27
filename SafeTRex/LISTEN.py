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
class Steer(Resource):
    def get(self, steer):
        getDriver().setAngle(int(steer))
        return {'steer': int(steer)}  


api.add_resource(Hello, '/hello')  # Route_1
api.add_resource(Speed, '/speed/<speed>')  # Route_3
api.add_resource(Steer, '/steer/<steer>')  # Route_3

if __name__ == '__main__':
    app.run(host= '0.0.0.0', port='5002')
