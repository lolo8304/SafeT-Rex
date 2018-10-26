from lane_recognition import *

def test():
  testL = Line(0, 360, 240, 0)
  testR = Line(480, 360, 240, 0)
  crossed, point = testL.line_intersection(testR)
  print("test intersection ", point)

  testL = Line(0, 360, 240, 0)
  testR = Line(480, 360, 480, 0)
  crossed, point = testL.line_intersection(testR)
  print("test intersection ", point)

  p = (240, 0)
  testDirectionX = steering_directionX(p, testL, testR, None, 480)
  print("test direct for ", p, " is ", testDirectionX)

  p = (-2147483600.0, -1994294800.0)
  testDirectionX = steering_directionX(p, testL, testR, None, 480)
  print("test direct for ", p, " is ", testDirectionX)

  p = (480, 0)
  testDirectionX = steering_directionX(p, testL, testR, None, 480)
  print("test direct for ", p, " is ", testDirectionX)

test()