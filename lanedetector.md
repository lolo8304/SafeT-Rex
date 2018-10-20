# Lane detector

# precondition
you need to have all python dependencies installed in your terminal or in a virtual environment.
We prefer using virtual environments




## running it on Pi3 with Picam


# test algorithm
we have recorded some sessions captured on the car that was moved by hand or with a remote control

## preconditions
- python3
- pip3

- opencv-python
- numpy
- atexit
- argparse
- collections
- time

## run on demo video

```bash
cd SafeTRex
python hough_transform.py --debug True

or

python hough_transform.py --debug True --config "axahack2018" --video ./lane_recognition/data/test_videos/demo-training-01.h264

```
default values for --config and --video


adapt pipeline via file hough_configuration.py
default configuration used "axahack2018" but can be configured using command line parameter --config

pipeline algorithms used in axahack2018
- crop (bottom) // to remove car
- crop (top) // currently 0 due to low camera position
- HLS_LUV_LAB // HLS, LUV, LAB channels 1, 0, 1 // draw result on this image
  based on https://www.linkedin.com/pulse/advanced-lane-finding-pipeline-tiba-razmi/
- GaussianBlur (keys to test settings: a, s to adjust "size", d, f to adjust "sigma")
- Grey
- Canny (keys to test settings: y, x to adjust "threshold1", c, v to adjust "threshold2", b, n to adjust "apertureSize")
- Hough
  not configurable: but implemented fix in code.
  use only detected lines with give slope
  use only detected lines where lines are close together
  display white (removed lines) and green (positive lines) in image

switched off pipelines for testing
- HLS_LUV_LAB_Threshold see above HLS_LUV_LAB but combining thresholds to 1 b/w channel
  https://www.linkedin.com/pulse/advanced-lane-finding-pipeline-tiba-razmi/
  still experimental
- Threshold (keys to test settings: 5, 6 to adjust "threshold", 7, 8 to adjust "max")
  still experimental, not very stable
- Warp // warp at bottom of image // assumption: lanes are not fully visible at bottom of image
  (keys to test settings: 1, 2 to adjust "warp_left", 3, 4 to adjust "warp_right")
  still experimental because of missinterpretation of cuttend warp lines on image for canny


## run on udacity recorded track

download from my dropbox https://www.dropbox.com/s/4v09dm11gk24cab/udacity-simulator_train-02.zip?dl=0 (74MB) and unpack in folder
in the folder there is a IMG where all the recorded images are stored (center* right* left*)

```bash
python hough_transform.py --debug True --imagesFolder <folder>/train-02/IMG/ --imagesPattern "center_" --config "udacity-simulator_track1"
```

checkout in ./lane_recognition/hough_configuration.py the configuration named "udacity_simulator_track1"

##images

![hackathon 2018 pipeline](/axahack2018-hough-train-demo.png)

![udacity recorded track pipeline](/udacity-hough-train-demo.png)

