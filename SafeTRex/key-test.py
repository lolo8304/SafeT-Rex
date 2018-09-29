import requests
import sys
import termios
import contextlib
from car_client import *

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--recording", required=False, type=int, default=0,
	help="number to write files accordinly")
args = vars(ap.parse_args())


@contextlib.contextmanager
def raw_mode(file):
    old_attrs = termios.tcgetattr(file.fileno())
    new_attrs = old_attrs[:]
    new_attrs[3] = new_attrs[3] & ~(termios.ECHO | termios.ICANON)
    try:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, new_attrs)
        yield
    finally:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, old_attrs)


def waitForKey(keys):
    print('exit with ^C or ^D')
    with raw_mode(sys.stdin):
        try:
            while True:
                ch = sys.stdin.read(1)
                if not ch or ch == chr(4):
                    break
                if ch in keys:
                  return ch
        except (KeyboardInterrupt, EOFError):
            pass


left = "j"
right = "l"
faster = "i"
slower = "m"
incSpeedFactor = "x"
decSpeedFactor = "y"
speedEdit = "e"
quit = "q"

driver = CarStateMachine(recording=args["recording"], init=-1, simulate=False)

if __name__ == '__main__':
  while(True):
    key = waitForKey(keys="jlimxyqse")
    if (key == left):
      driver.left()
    elif key == right:
      driver.right()
    elif key == faster:
      driver.faster()
    elif key == slower:
      driver.slower()
    elif key == incSpeedFactor:
      driver.incSpeedFactor()
    elif key == decSpeedFactor:
      driver.decSpeedFactor()
    elif key == speedEdit:
      speed = input("Enter your speed (0-100)? ")
      driver.setRUN(int(speed))
    elif key == quit:
      driver.setRUN(0)
      driver.setAngle(0)
      break
