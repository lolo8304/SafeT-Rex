import picamera
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--len", required=True, type=int,
	help="len of video")
ap.add_argument("-f", "--format", required=False, default="h264",
	help="video format")
ap.add_argument("-e", "--ext", required=False, default="",
	help="extension")
args = vars(ap.parse_args())


with picamera.PiCamera() as camera:
    camera.resolution = (640, 480)
    filename = 'safet-rex.'+args["format"]
    if args["ext"] != "":
        filename = filename + "." + ext

    print ('recording ',args["len"], 's and save to ',filename)

    camera.start_recording(filename)
    camera.wait_recording(args["len"])
    camera.stop_recording()
    print("... done")