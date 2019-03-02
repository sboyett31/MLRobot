'''
This is where the Serial data will be received from the image processing.
'''

import time
import serial
import random as r

x_speed = -(r.randint(5, 10))


def recv_data():
    # Work with image processing on this.
    x = r.randint(-100,100)
    y = r.randint(0, 100)
    return x, y


def recv_dummy_data():
    # Generates dummy data of a puck headed straight toward the robot
    x_start = r.randint(25, 100)
    y_start = r.randint(0, 100)
    y_speed = 0
    return x_start, y_start, x_speed, y_speed

