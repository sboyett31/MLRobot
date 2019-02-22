'''
This is where the Serial data will be received from the image processing.
'''

import time
import serial
import random as r


def recv_data():
    # Work with image processing on this.
    x = r.randint(-100,100)
    y = r.randint(0, 100)
    return x, y