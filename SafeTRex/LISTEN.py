from flask import Flask
from flask_restful import Resource, Api
from car import

app = Flask(__name__)
api = Api(app)


class Hello(Resource):
    def get(self):
        return {'hello back': "hello"}  # Fetches first column that is Employee ID


class Speed(Resource):
    def get(self, speed):
        return {'speed': speed}  # Fetches first column that is Employee ID


api.add_resource(Hello, '/hello')  # Route_1
api.add_resource(Speed, '/speed/<speed>')  # Route_3

if __name__ == '__main__':
    app.run(host= '0.0.0.0', port='5002')
