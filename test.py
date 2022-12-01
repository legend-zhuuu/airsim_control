import math
import cmath
import airsim
import time
import numpy as np
from socket import socket, AF_INET, SOCK_STREAM
from functools import partial
import cv2

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def rect_path(round, length, height, radius, step=50):
    path = []
    for r in range(round):
        path_line1 = [airsim.Vector3r(i*length/step, -4*radius*r, height) for i in range(step+1)]
        path_round1 = [airsim.Vector3r(length + radius*(math.sin(theta*math.pi/step)), -4*radius*r - (radius*(1 - math.cos(theta*math.pi/step))), height)
                       for theta in range(step)]
        path_line2 = [airsim.Vector3r(length - i*length/step, -4*radius*r - 2*radius, height) for i in range(step + 1)]
        path_round2 = [airsim.Vector3r(-radius*(math.sin(theta*math.pi/step)), -4*radius*r - 2*radius - (radius*(1 - math.cos(theta*math.pi/step))), height)
                       for theta in range(step)]
        path = path + path_line1 + path_round1 + path_line2 + path_round2
    path_line1 = [airsim.Vector3r(i * length / step, -4 * radius * round, height) for i in range(step + 1)]
    path_round1 = [airsim.Vector3r(length + radius*(math.sin(theta*math.pi/step)), -4*radius*round - (radius*(1 - math.cos(theta*math.pi/step))), height)
                       for theta in range(step)]
    path_line2 = [airsim.Vector3r(length - i * length / step, -4 * radius * round - 2 * radius, height) for i in range(step + 1)]
    path_round2 = [airsim.Vector3r(length + radius * (math.sin(theta * math.pi / step / 2)), -4 * radius * round + (radius * (1 - math.cos(theta * math.pi / step / 2))), height)
                    for theta in range(step)]
    start = (0, 0)
    end = (length + radius, -4 * radius * round + radius)
    path_back = [airsim.Vector3r((end[0] - start[0]) * (step - i)/step, (end[1] - start[1]) * (step - i)/step, height) for i in range(step + 1)]
    path = path + path_line1 + path_round2 + path_back
    return path

#
# path = rect_path(5, 600, -200, 30)
# x = []
# y = []
# for point in path:
#     x.append(point.x_val)
#     y.append(point.y_val)


# plt.subplot(111)
# N = 12
# theta = np.linspace(0, N * np.pi, 1000)
# a = 1
# b = 10
# x = (a * b*theta) * np.cos(theta)
# y = (a * b * theta) * np.sin(theta)
# start = [x[0], y[0]]
# end = [x[-1], y[-1]]
# dist = math.sqrt((start[0]-end[0])**2+(start[1]-end[1])**2)
# theta = np.linspace(0, np.pi, 1000)
# x2 = dist/2 + dist/2*np.cos(theta)
# y2 = dist/2*np.sin(theta)
# x = np.append(x, x2)
# y = np.append(y, y2)
# plt.plot(x, y)
# plt.plot()
# plt.show()


pic = cv2.imread(r'C:\Users\1328301164\Documents\1645363805184949504.png', flags=2)
pic[pic<1] = 120*255
print(pic/255)
# depth_int = []
# for i in pic[788:1088]:
#     for j in i[732:1032]:
#         depth_int.append(math.ceil(j/255))
# x = dict((a, depth_int.count(a)) for a in depth_int)
# y = [k for k, v in x.items() if max(x.values()) == v]
# print(y)
print(np.min(pic/255))
print(np.max(pic/255))

