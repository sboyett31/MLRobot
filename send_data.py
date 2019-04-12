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

    if 0 <= int(data) <= 300:
        line = str(ser.readline())
        if "not ready" not in line:
            ser.write(data.encode(encoding='ascii'))
            print("Sent " + data)
        elif "not ready" in line:
            print("(E) Hockey stick is currently moving... data not sent.")
    else:
        print("(E) Data out of range: {} - Not sent to Arduino.".format(data))
