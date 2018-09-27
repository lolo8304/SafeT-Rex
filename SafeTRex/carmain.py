from car import *
import time
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--speed", required=False, type=int, default=False,
	help="speed mode")
ap.add_argument("-t", "--stop", required=False, type=int, default=False,
	help="speed mode")
ap.add_argument("-c", "--steer", required=False, type=int, default=False,
	help="steer mode")
args = vars(ap.parse_args())

#__driver = CarStateMachine()
__car = ServoCar()
GPIO.setwarnings(False)

if (args["speed"] is not None):
  #__driver.setRUN(args["speed"])
  __car.speed(args["speed"])

if (args["steer"] is not None):
  #__driver.setAngle(args["steer"])
  __car.steer(args["steer"])

if (args["stop"] is not None):
  pass
  #__driver.setSTOP()

print("sleep to close")
time.sleep(2)
__car.stop()