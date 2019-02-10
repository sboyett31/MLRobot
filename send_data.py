'''
This is where data will be sent to the arduino to control our robot.
'''

import time
import serial

def main():
    # Parameters: COM PORT, Baud Rate, Timeout
    ser = serial.Serial(3, 9600, timeout=1)
    print("Connection Established.")
    # sending z accross the serial connection
    ser.write('z'.encode("ascii"))
    ser.close()
