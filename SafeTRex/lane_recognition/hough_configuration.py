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
        "draw" : True,
        "parameters" : {
          "algorithm" : "bottom",
          "factor" : 0.43333333
        }
      },
      {
        "type" : "Grey",
        "parameters" : {
        }
      },
      {
        "type" : "GaussianBlur",
        "parameters": {
          "size" : 23,
          "sigma" : -0.1
        },
        "keys" : {
          "a" : {
            "name" : "size",
            "f": (lambda params: params["size"] - 2)
          },
          "s" : {
            "name" : "size",
            "f" : (lambda params: params["size"] + 2)
          },
          "d" : {
            "name" : "sigma",
            "f": (lambda params: params["sigma"] - 0.1)
          },
          "f" : {
            "name" : "sigma",
            "f" : (lambda params: params["sigma"] + 0.1)
          }
        }
      },
      {
        "type" : "Canny",
        "parameters": {
          "threshold1" : 101.0,
          "threshold2" : 101.0,
          "apertureSize" : 3,
          "L2gradient" : False
        },
        "keys" : {
          "y" : {
            "name" : "threshold1",
            "f": (lambda params: params["threshold1"] - 2.0)
          },
          "x" : {
            "name" : "threshold1",
            "f" : (lambda params: params["threshold1"] + 2.0)
          },
          "c" : {
            "name" : "threshold2",
            "f": (lambda params: params["threshold2"] - 2.0)
          },
          "v" : {
            "name" : "threshold2",
            "f" : (lambda params: params["threshold2"] + 2.0)
          },
          "b" : {
            "name" : "apertureSize",
            "f": (lambda params: params["apertureSize"] - 2)
          },
          "n" : {
            "name" : "apertureSize",
            "f" : (lambda params: params["apertureSize"] + 2)
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
          "factor" : 0.33333333
        },
      },
      {
        "type" : "Crop",
        "draw" : True,
        "parameters" : {
          "algorithm" : "bottom",
          "factor" : 0.25,
        },
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
            "f": (lambda params: params["size"] - 2)
          },
          "e" : {
            "name" : "size",
            "f" : (lambda params: params["size"] + 2)
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
            "f": (lambda params: params["size"] - 2)
          },
          "e" : {
            "name" : "size",
            "f" : (lambda params: params["size"] + 2)
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
            "f": (lambda params: params["kernel"] - 2)
          },
          "s" : {
            "name" : "kernel",
            "f" : (lambda params: params["kernel"] + 2)
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
            "f": (lambda params: params["size"] - 2)
          },
          "s" : {
            "name" : "size",
            "f" : (lambda params: params["size"] + 2)
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
            "f": (lambda params: params["threshold"] - 2)
          },
          "s" : {
            "name" : "threshold",
            "f" : (lambda params: params["threshold"] + 2)
          }
        }
      },
      {
        "type" : "Grey",
        "parameters" : {
        },
      },
      {
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
            "f": (lambda params: params["threshold1"] - 2.0)
          },
          "x" : {
            "name" : "threshold1",
            "f" : (lambda params: params["threshold1"] + 2.0)
          },
          "c" : {
            "name" : "threshold2",
            "f": (lambda params: params["threshold2"] - 2.0)
          },
          "v" : {
            "name" : "threshold2",
            "f" : (lambda params: params["threshold2"] + 2.0)
          }
        }
      }
    ],
    "title": "udacity simulation track 1",
    "description" : "simulated track from udacity - track 1"
  }

