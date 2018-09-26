from SafeTRex.main import *
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--debug", required=False, type=bool, default=False,
	help="debug mode")
args = vars(ap.parse_args())


print("Hello World")
Handler = CarHandler(args)
Handler.start()

