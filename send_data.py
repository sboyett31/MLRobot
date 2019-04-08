'''
This is where data will be sent to the arduino to control our robot.
'''

import serial
import time


def est_serial_cnxn():
    try:
        ser = serial.Serial("/dev/ttyACM0", 9600, timeout=10)
        if ser.is_open:
            print("Connection ACM0 opened..")
    except:
        ser = serial.Serial("/dev/ttyACM1", 9600, timeout=10)
        if ser.is_open:
            print("Connection ACM1 opened..")
    return ser


def send(ser, data):

    ser.write(data.encode(encoding='ascii'))
    print("Sent " + data)