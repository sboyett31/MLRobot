'''
This is where the Serial data will be received from the image processing.
'''
import socket
from socket import timeout 
import random as r


def est_TCP_cnxn():
    TCP_IP = "192.168.0.2"
    TCP_PORT = 5000
    print("Connecting....")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))
    s.settimeout(1)
    print("Connection established!")
    return s

def recv_puck(tcp):
    # Receives x and y data from image processing PI.
    rdymsg = "ready"
    tcp.send(rdymsg.encode())
    try:
        MESSAGE = tcp.recv(8)
        MESSAGE = MESSAGE.decode()
        #print(MESSAGE)
    except timeout:
        # Realize that a goal is scored after no data is received for a sec
        return -999, -999
    if MESSAGE != "close": 
        x = int(MESSAGE.split(',')[0])
        y = int(MESSAGE.split(',')[1])
    
        return -x,y
    elif MESSAGE == "close":
        tcp.close()
        return -998, -988


def recv_robo(ser):
    if ser.in_waiting:
        line = str(ser.readline())
        #print("Received line: {}".format(line))
        line = line.split("'")[1]
        line = line.split("\\")[0]
        #print("Received robot position: {}".format(line))

        if line:
            robot_pos = int(line)/9
            robot_pos = int(robot_pos)
        else: robot_pos = None
    else:
        robot_pos = None

    return robot_pos


def recv_data(tcp, ser):
    # Receives x and y data from image processing.
    # s.send(input("MSG: ").encode())
    rdymsg = "ready"
    tcp.send(rdymsg.encode())
    try:
        MESSAGE = tcp.recv(8)
        MESSAGE = MESSAGE.decode()                          
        # print(MESSAGE)
    except timeout:
        # Realize that a goal is scored if no data is received after a second
        return -999, -999, -999
    
    line = str(ser.readline())
    print("Received line: {}".format(line))
    line = line.split("'")[1]
    line = line.split("\\")[0]
   
    if line:
        robot_pos = int(line)/9
    else:
        robot_pos = 0

    if MESSAGE != "close": 
        x = int(MESSAGE.split(',')[0])
        y = int(MESSAGE.split(',')[1])
    
        return robot_pos, x,y

    elif MESSAGE == "close":
        tcp.close()
        return -998, -998, -988


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

