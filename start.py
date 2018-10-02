from SafeTRex.main import *

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--debug", required=False, type=bool, default=False,
	help="debug mode")
ap.add_argument("-x", "--xdebug", required=False, type=bool, default=False,
	help="X debug mode")
ap.add_argument("-r", "--recording", required=False, type=int, default=0,
	help="number to write files accordinly")
args = vars(ap.parse_args())


print("Hello World")
Handler = CarHandler(args)
Handler.start()

