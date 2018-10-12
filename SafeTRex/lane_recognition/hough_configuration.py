def configurations():
  return {
    "axahack2018" : axa_hackathon_lane(),
    "udacity-simulator_track1" : udacity_simulator_track1()
  }


def axa_hackathon_lane():
  return {
    "pipeline" : [
      {
        "type" : "Crop",
        "parameters" : {
          "algorithm" : "bottom",
          "factor" : 0.4
        }
      },
      {
        "type" : "Crop",
        "parameters" : {
          "algorithm" : "top",
          "factor" : 0.0
        }
      },
      {
        "type" : "HLS_LUV_LAB",
        "draw" : True,
        "parameters": {
          "channel1" : 1,
          "channel2" : 0,
          "channel3" : 1,
          "threshold" : 128,
          "max" : 255,
          "type" : 0
        }
      },
      {
        "type" : "GaussianBlur",
        "parameters": {
          "size" : 9,
          "sigma" : -0.1
        },
        "keys" : {
          "a" : {
            "name" : "size",
            "f": (lambda value : value - 2)
          },
          "s" : {
            "name" : "size",
            "f" : (lambda value : value + 2)
          },
          "d" : {
            "name" : "sigma",
            "f": (lambda value : value - 0.1)
          },
          "f" : {
            "name" : "sigma",
            "f" : (lambda value : value + 0.1)
          }
        }
      },
      {
        "type" : "Grey",
        "off" : False,
        "parameters" : {
        },
      },
      {
        "type" : "Threshold",
        "off" : True,
        "parameters": {
          "threshold" : 128,
          "max" : 255,
          "type" : 0
        },
        "keys" : {
          "5" : {
            "name" : "threshold",
            "f": (lambda value : value - 2)
          },
          "6" : {
            "name" : "threshold",
            "f" : (lambda value : value + 2)
          },
          "7" : {
            "name" : "max",
            "f": (lambda value : value - 2)
          },
          "8" : {
            "name" : "max",
            "f" : (lambda value : value + 2)
          }
        }
      },
      {
        "type" : "Warp",
        "off" : True,
        "parameters": {
          "warp_left" : 195,
          "warp_right" : 395,
        },
        "keys" : {
          "1" : {
            "name" : "warp_left",
            "f": (lambda value : value - 2)
          },
          "2" : {
            "name" : "warp_left",
            "f" : (lambda value : value + 2)
          },
          "3" : {
            "name" : "warp_right",
            "f": (lambda value : value - 2)
          },
          "4" : {
            "name" : "warp_right",
            "f" : (lambda value : value + 2)
          }
        }
      },
      {
        "type" : "Canny",
        "parameters": {
          "threshold1" : 43.0,
          "threshold2" : 121.0,
          "apertureSize" : 3,
          "L2gradient" : True
        },
        "keys" : {
          "y" : {
            "name" : "threshold1",
            "f": (lambda value : value - 2.0)
          },
          "x" : {
            "name" : "threshold1",
            "f" : (lambda value : value + 2.0)
          },
          "c" : {
            "name" : "threshold2",
            "f": (lambda value : value - 2.0)
          },
          "v" : {
            "name" : "threshold2",
            "f" : (lambda value : value + 2.0)
          },
          "b" : {
            "name" : "apertureSize",
            "f": (lambda value : value - 2)
          },
          "n" : {
            "name" : "apertureSize",
            "f" : (lambda value : value + 2)
          }
        }
      }
    ],
    "title": "AXA hackathon",
    "description" : "real track with raspi on roof, tilted, yellow lines, black floor"
  }



def udacity_simulator_track1():
  return {
    "pipeline" : [
      {
        "type" : "Crop",
        "parameters" : {
          "algorithm" : "top",
          "factor" : 0.43333333
        },
      },
      {
        "type" : "Crop",
        "parameters" : {
          "algorithm" : "bottom",
          "factor" : 0.25,
        },
      },
      {
        "type" : "HLS_LUV_LAB",
        "draw" : True,
        "parameters": {
          "channel1" : 1,
          "channel2" : 0,
          "channel3" : 1
        }
      },
      {
        "type" : "medianBlur",
        "off" : False,
        "parameters": {
          "size" : 9
        },
        "keys" : {
          "w" : {
            "name" : "size",
            "f": (lambda value : value - 2)
          },
          "e" : {
            "name" : "size",
            "f" : (lambda value : value + 2)
          }
        }
      },
      {
        "type" : "GaussianBlur",
        "off" : True,
        "parameters": {
          "size" : 17,
          "sigma" : 0
        },
        "keys" : {
          "w" : {
            "name" : "size",
            "f": (lambda value : value - 2)
          },
          "e" : {
            "name" : "size",
            "f" : (lambda value : value + 2)
          }
        }
      },      {
        "type" : "SelectiveGaussianBlur",
        "off" : True,
        "parameters": {
          "kernel" : 17
        },
        "keys" : {
          "a" : {
            "name" : "kernel",
            "f": (lambda value : value - 2)
          },
          "s" : {
            "name" : "kernel",
            "f" : (lambda value : value + 2)
          }
        }
      },
      {
        "type" : "AdaptiveThreshold",
        "off" : True,
        "parameters": {
          "max" : 5.0,
          "method" : 0,
          "size" : 5
        },
        "keys" : {
          "a" : {
            "name" : "size",
            "f": (lambda value : value - 2)
          },
          "s" : {
            "name" : "size",
            "f" : (lambda value : value + 2)
          }
        }
      },
      {
        "type" : "Threshold",
        "off" : True,
        "parameters": {
          "threshold" : 50,
          "max" : 255,
          "type" : 1
        },
        "keys" : {
          "a" : {
            "name" : "threshold",
            "f": (lambda value : value - 2)
          },
          "s" : {
            "name" : "threshold",
            "f" : (lambda value : value + 2)
          }
        }
      },
      {
        "type" : "Grey",
        "parameters" : {
        },
      },
      {
        "type" : "Warp",
        "off" : True,
        "parameters": {
          "warp_left" : 137,
          "warp_right" : 169,
        },
        "keys" : {
          "1" : {
            "name" : "warp_left",
            "f": (lambda value : value - 2)
          },
          "2" : {
            "name" : "warp_left",
            "f" : (lambda value : value + 2)
          },
          "3" : {
            "name" : "warp_right",
            "f": (lambda value : value - 2)
          },
          "4" : {
            "name" : "warp_right",
            "f" : (lambda value : value + 2)
          }
        }
      },      {
        "type" : "Canny",
        "parameters": {
          "threshold1" : 65.0,
          "threshold2" : 65.0,
          "apertureSize" : 3,
          "L2gradient" : False
        },
        "keys" : {
          "y" : {
            "name" : "threshold1",
            "f": (lambda value : value - 2.0)
          },
          "x" : {
            "name" : "threshold1",
            "f" : (lambda value : value + 2.0)
          },
          "c" : {
            "name" : "threshold2",
            "f": (lambda value : value - 2.0)
          },
          "v" : {
            "name" : "threshold2",
            "f" : (lambda value : value + 2.0)
          }
        }
      }
    ],
    "title": "udacity simulation track 1",
    "description" : "simulated track from udacity - track 1"
  }

