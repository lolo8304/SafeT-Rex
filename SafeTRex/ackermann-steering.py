import math
from lane_recognition.Line import Line

class car_steering:
    def __init__(self, L, T):

        self.L = np.float32(L)
        self.y1 = np.float32(T)

    def r(self, delta):
      pass


def createLine(xy, wh):
  return Line (xy[0], xy[1]+wh[1], xy[0]+wh[0], xy[1])

def degrees(xy1, wh1, xy2, wh2):
  back20 = createLine(xy1, wh1)
  front20 = createLine(xy2, wh2)
  print ("degree = ", back20.degree)
  print ("degree = ", front20.degree)
  print(back20.lamdaF())
  print(front20.lamdaF())
  return front20.degree - back20.degree


back20_xy = (4.74,1.36)
back20_wh = (16.22, 13.5)
front20_xy = (4.74, 1.36)
front20_wh = (16.42, 10.89)

print(degrees(back20_xy, back20_wh, front20_xy, front20_wh))

back15_xy = (12.28, 6.36)
back15_wh = (8.71, 12.69)
front15_xy = (7.43, -0.15)
front15_wh = (10.01, 12.25)

print(degrees(back15_xy, back15_wh, front15_xy, front15_wh))


back10_xy = (9.62, 10.52)
back10_wh = (3.14, 9.71)
front10_xy = (6.17, 1.46)
front10_wh = (3.79, 8.85)

print(degrees(back10_xy, back10_wh, front10_xy, front10_wh))
