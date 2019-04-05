'''
This is where the Serial data will be received from the image processing.
'''


import socket
import random as r


def recv_data():
    # Receives x and y data from image processing.  Returns flag when new data is present.
    TCP_IP = socket.gethostname()
    TCP_PORT = 5000
    BUFFER_SIZE = 1024
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))
    while 1:
        # s.send(input("MSG: ").encode())
        MESSAGE = s.recv(256)
        print(MESSAGE)
        # data = s.recv(BUFFER_SIZE)

    s.close()

    x = r.randint(-100, 100)
    y = r.randint(0, 100)
    return updated, x, y


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

