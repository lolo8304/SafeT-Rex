import time
import numpy as np


def find_longest_none_zero(array, size):
    # example [0 [obj] 0 0 0 0 0 0 [obj] [obj] [obj,obj] [obj,obj,obj] 0 [obj] [obj,obj] ]
    cc = np.zeros([size], int)
    i = 0
    maxC = 0
    maxI = -1
    minI = 0
    minMaxI = 0
    for a in array:
      if i > 0 and len(a) > 0:
          if cc[i-1] == 0:
            minI = i
          cc[i] = cc[i-1] + len(a)
      else:
          cc[i] = len(a)
      if cc[i] > maxC:
        maxC = cc[i]
        maxI = i
        minMaxI = minI
      i = i+1
    return minMaxI, maxI, maxC

def keep_longest_non_zero(array, size):
  minIndex, maxIndex, max = find_longest_none_zero(array, size)
  cc = []
  for i in range(minIndex, maxIndex+1):
    cc.extend(array[i])
  return cc

def count_distance(array, f, max, distance):
    max_index = max // distance + 1
    count = [[] for _ in range(max_index)]
    print("empty",count)
    for a in array:
        i = int( f(a) // distance)
        count[i].append(a)
    return count

def keep_closests(array, f, max, distance):
    cc = count_distance(array, f, max, distance)
    return keep_longest_non_zero(cc, len(cc))


def fx(a):
  return a[0]

a = [ (1,2), (2,3), (10, 3), (9,1),(9,2),(10,3),  (3,3)]
print(count_distance(a, fx, 10, 2))

print(keep_closests(a, fx, 10, 2))