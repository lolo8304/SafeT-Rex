from car import *
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--speed", required=False, type=int, default=False,
	help="speed mode")
ap.add_argument("-t", "--stop", required=False, type=int, default=False,
	help="speed mode")
ap.add_argument("-c", "--steer", required=False, type=int, default=False,
	help="steer mode")
args = vars(ap.parse_args())

__driver = CarStateMachine()

if (args["speed"] > 0):
  __driver.setRUN(args["speed"])

if (args["steer"] > 0):
  __driver.setAngle(args["steer"])

if (args["stop"] > 0):
  __driver.setSTOP()

