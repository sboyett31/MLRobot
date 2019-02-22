from recv_data import recv_data
from actions import hit_puck, move_up, move_down


class AirHockeyEnv:
    def __init__ (self):
        self._robot_pos = 0
        self._prev_x = 0
        self._prev_y = 0
        self._puck_x = 0
        self._puck_y = 0
        self._speed_x = 0
        self._speed_y = 0

    def reset(self):
        # This function will be responsible for resetting the environment
        # When necessary. (Primarily robot position)
        new_state = []
        rc = recv_data()
        move_down(100)        # move down to y = 0
        self._robot_pos = 50        # not sure when to implement this, could do move_up(50)
        self._puck_x = rc[0]
        self._puck_y = rc[1]
        self._speed_x = 0
        self._speed_y = 0

        new_state.append(self._robot_pos)
        for x in range(2): new_state.append(rc[x])
        new_state.append(self._speed_x)
        new_state.append(self._speed_y)

        return new_state

    def update(self):
        # This function will receive values from image processing to update
        # environment.

        new_state = []
        self._prev_x = self._puck_x
        self._prev_y = self._puck_y
        rc = recv_data()

        # update values based on data returned by receive data function
        self._puck_x = rc[0]
        self._puck_y = rc[1]
        self._speed_x = self.get_speed('x')
        self._speed_y = self.get_speed('y')

        new_state.append(self._robot_pos)
        for x in range(2): new_state.append(rc[x])
        new_state.append(self._speed_x)
        new_state.append(self._speed_y)

        return new_state

    def step(self, action):
        # This function will be responsible for stepping to the next state
        # Based on the action chosen by the neural network
        # This could use the algorithm joel wrote to give some sort of varying reward

        done = 0
        env_reward = 0
        new_state = []
        prev_speed_x = self._speed_x
        self._prev_x = self._puck_x
        self._prev_y = self._puck_y

        if action.startswith("move up"):
            intensity = action.split(' ')[2]
            move_up(intensity)
            # OR (Based on how we move the robot physically
            # This part would just update robot_pos var
            # Then wait for certain time to send data to arduino
            self._robot_pos += intensity
        elif action.startswith("move down"):
            intensity = action.split(' ')[2]
            self._robot_pos -= intensity
            move_down(intensity)
            # Or
            self._robot_pos -= intensity
        elif action.startswith("hit puck"):
            direction = action.split(' ')[2]
            hit_puck(direction)
        elif action == "nothing":
            pass

            rc = recv_data()
            self._puck_x = rc[0]
            self._puck_y = rc[1]
            self._speed_x = self.get_speed('x')
            self._speed_y = self.get_speed('y')

            # Determine a reward for action taken
            if self._robot_pos == self._puck_y and self._puck_x < 0:
                # Puck is close and robot is mirroring it
                env_reward += 10
            elif self._robot_pos != self._puck_y and self._puck_x < 0:
                # Puck is close and robot is not mirroring it
                env_reward -= abs(self._robot_pos - self._puck_y)

            if self._puck_x == self._puck_y == -999:
                done = 1
                if prev_speed_x > 0:
                    # Puck was going in positive direction ( away from our robot )
                    env_reward = 100

            new_state.append(self._robot_pos)
            for x in range(2): new_state.append(rc[x])
            new_state.append(self._speed_x)
            new_state.append(self._speed_y)

        return new_state, env_reward, done

    def position_robot(self):
        # One strategy is to use this function and wait to call it until puck is approaching
        # Either way, robot pos will be updated dynamically while puck is moving
        move_up(self._robot_pos)
        return

    def in_play(self):
        # Return true if puck is in play
        return self._puck_x != -999 and self._puck_y != -999

    def get_speed(self, axis):
        # returns the speed based on a previous position (v1) and a current position (v2)
        v1 = self._prev_x, v2 = self._puck_x if axis == 'x' else self._prev_y, self._puck_y
        if self.in_play():
            if v2 < v1:
                return -abs(v2 - v1)
            elif v2 > v1:
                return abs(v2 - v1)
        elif not self.in_play():
            return 0
