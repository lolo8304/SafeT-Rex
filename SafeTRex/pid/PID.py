import time
import sys


class PID(object):

  def __init__(self, Kp = 0.0, Ki = 0.0, Kd = 0.0):
    self.error_proportional = 0.0
    self.error_integral = 0.0
    self.error_derivative= 0.0
    self.Kp = Kp;
    self.Ki = Ki;
    self.Kd = Kd;

  def updateError(self, cte):
    self.error_integral += cte
    self.error_derivative = cte - self.error_proportional
    self.error_proportional = cte

  def TotalError(self) {
    return -(self.Kp * self.error_proportional + self.Ki_ * self.error_integral + self.Kd * self.error_derivative)

