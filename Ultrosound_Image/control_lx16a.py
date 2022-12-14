import sys, time
from math import sin, cos
from lx16a import *

LX16A.initialize("/dev/ttyUSB0")
servo1 = LX16A(1)

servo1.angleOffsetAdjust(18)
servo1.angleOffsetWrite()

angle  = float(sys.argv[1])

start = time.time()

while(True):
    servo1.moveTimeWrite(angle, 1)
    true_angle = servo1.getPhysicalPos()
    if abs(true_angle - angle) < 1:
        break
    else:
        servo1.moveTimeWrite(angle+1, 1)
    end = time.time()
    if (end-start) > 5.0:
        print("######################## Timeout ########################\n")
        break


