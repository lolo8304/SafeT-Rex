import picamera
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--len", required=True, type=int,
	help="len of video")
ap.add_argument("-f", "--format", required=False, default="h264",
	help="video format")
args = vars(ap.parse_args())


with picamera.PiCamera() as camera:
    camera.resolution = (640, 480)
    filename = 'saft-rex.'+args["format"]
    print ('recording ',args["len"], 's and save to ',filename)

    camera.start_recording(filename)
    camera.wait_recording(args["len"])
    camera.stop_recording()
    print("... done")