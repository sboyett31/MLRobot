'''
This is where the Serial data will be received from the image processing.
'''


import socket
import random as r


def est_cnxn():
    TCP_IP = socket.gethostname()
    TCP_PORT = 5000
    BUFFER_SIZE = 1024
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))
    return s

def recv_data(s):
    # Receives x and y data from image processing.  Returns flag when new data is presents
    # s.send(input("MSG: ").encode())
    BUFFER_SIZE = 1024
    MESSAGE = s.recv(1024)
    print(MESSAGE)

    x = MESSAGE.split(',')[0]
    y = MESSAGE.split(',')[1]

    return x, y

def close_cnxn(s):
    s.close()

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

