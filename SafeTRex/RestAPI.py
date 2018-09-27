import requests
import socket
import sys
from urllib.parse import parse_qs, urlparse
from .car import ServoCar
url = 'http://httpbin.org/get'
MSGLEN = 1000
def get_steer(url):
    try:
        return parse_qs(urlparse(url).query(["steer"]))
    except KeyError:
        return []

def get_speed(url):
    try:
        return parse_qs(urlparse(url).query(["speed"]))
    except KeyError:
        return []
HOST = ""
PORT = 8080

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(HOST, PORT)
    s.listen()
    conn, addr = s.accept()



payload = {}

socket = MySocket()
socket.bind<
socket.connect("","8080")

while True:

    r = requests.get(url, params=payload)
    car = ServoCar()
    car.speed(get_speed(r.url))
    car.steer(get_steer(r.url))
