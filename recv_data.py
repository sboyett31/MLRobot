'''
This is where the Serial data will be received from the image processing.
'''

import time
import serial
import random as r


def recv_data():
    # Work with image processing on this.
    x = r.randint(-100, 100)
    y = r.randint(0, 100)
    return x, y


def recv_dummy_data():
    # Generates dummy data of a puck headed straight toward the robot
    x_speed = -(r.randint(1, 5))
    y_speed_down = -(r.randint(1, 5))
    y_speed_up = (r.randint(1, 5))
    y_dir = r.randint(1, 2)

    if y_dir == 1:
        y_speed = y_speed_down
    elif y_dir == 2:
        y_speed = y_speed_up

    # y_speed = 0

    # print("Y speed is: {}".format(y_speed))
    # x_start = r.randint(25, 100)
    x_start = 100                   # Always send x from end
    y_start = r.randint(0, 100)

    return x_start, y_start, x_speed, y_speed

