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

