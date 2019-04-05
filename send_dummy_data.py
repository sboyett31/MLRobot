import socket
import random as r
from time import sleep



def gen_dummy_data():
    #Generates dummy data of puck

    x_speed = -(r.randint(1, 5))
    y_speed_down = -(r.randint(1, 5))
    y_speed_up = (r.randint(1, 5))
    y_dir = r.randint(1, 2)
    if y_dir == 1:
        y_speed = y_speed_down
    elif y_dir == 2:
        y_speed = y_speed_up

    x_start = 100
    y_start = r.randint(0, 100)

    return x_start, y_start, x_speed, y_speed


TCP_IP = socket.gethostname()
TCP_PORT = 5000
BUFFER_SIZE = 1024
MESSAGE = "Hello, World!"

data_out = []

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))

data_out[0], data_out[1], data_out[2], data_out[3] = gen_dummy_data()


while 1:

    s.send(input(data_out[0]).encode())
    #data = s.recv(BUFFER_SIZE)

s.close()

print ("received data:", data)