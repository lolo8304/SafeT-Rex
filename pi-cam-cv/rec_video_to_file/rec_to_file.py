import picamera
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--len", required=True, type=int,
	help="len of video")
args = vars(ap.parse_args())


with picamera.PiCamera() as camera:
    camera.resolution = (640, 480)
    camera.start_recording('my_video.h264')
    camera.wait_recording(args["len"])
    camera.stop_recording()