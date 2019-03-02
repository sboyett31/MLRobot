'''
Neural Network will have outputs that will be either 1 or 0.
Based on the outputs, These functions will be called.
For example: The hitpuck_5 output will call the hitpuck function with an intensity of 5.
'''

from send_data import send


def hit_puck(direction):
    # spin hockey stick in direction specified
    direction = direction.split('')[0]
    send("h{}".format(direction))


def move_up(intensity):
    # move robot left based on intensity
    send("u{}".format(intensity))


def move_down(intensity):
    # move robot right based on intensity
    send("d{}".format(intensity))

