from car_client import *
import time
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--speed", required=False, type=int,
	help="speed mode")
ap.add_argument("-t", "--stop", required=False, type=int,
	help="speed mode")
ap.add_argument("-c", "--steer", required=False, type=int,
	help="steer mode")
args = vars(ap.parse_args())

__driver = CarStateMachine()
#__car = ServoCar()

if (args["speed"] is not None):
  __driver.setRUN(args["speed"])
  #__car.speed(args["speed"])

if (args["steer"] is not None):
  __driver.setAngle(args["steer"])
  #__car.steer(args["steer"])

if (args["stop"] is not None):
  pass
  #__driver.setSTOP()

#__car.stop()
__driver.close()