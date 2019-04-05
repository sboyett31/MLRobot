from recv_data import recv_data, recv_dummy_data
from send_data import send
import os
from actions import hit_puck, move_up, move_down
from constants import X_MAX, X_MIN, Y_MAX, Y_MIN
import numpy as np
import time
import gym
from gym.utils import seeding

class AirHockeyEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    def __init__ (self):
        # Send to arduino actual Y position, not increments
        self._robot_pos = 0
        self._prev_x = 0
        self._prev_y = 0
        self._puck_x = 0
        self._puck_y = 0
        self._speed_x = 0
        self._speed_y = 0
        self.hit_up = False
        self.hit_down = False

        # Definining variables for rendering environment
        self.top_wall = None
        self.bot_wall = None
        self.goal = None
        self.puck = None
        self.stick = None
        self.viewer = None
        self.puck_trans = None
        self.stick_trans = None

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # This function will be responsible for resetting the environment
        # When necessary. (Primarily robot position)
        new_state = []
        # rc = recv_data()
        rc = recv_dummy_data()
        # move_down(100)              # move down to y = 0
        self._robot_pos = 50        # not sure when to implement this, could do move_up(50)
        self._puck_x = rc[0]
        self._puck_y = rc[1]
        self._speed_x = rc[2]
        self._speed_y = rc[3]

        new_state.append(self._robot_pos)
        for x in range(4): new_state.append(rc[x])

        return np.array(new_state)

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 400
        world_width = 200
        world_height = 100
        scale = 4
        clearance = 10

        pr = 15  # puck radius

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            self.top_wall = rendering.Line(start=(0, 395), end=(200*scale, 395))
            self.bot_wall = rendering.Line(start=(0, 0), end=(200*scale, 0))

            self.goal = rendering.Line(start=(1.0, 40.0*scale), end=(1.0, 60.0*scale))
            self.stick = rendering.Line(start=(8, (self._robot_pos*scale) - 20), end=(8, (self._robot_pos*scale) + 20))
            self.puck = rendering.make_circle(radius=pr, res=30, filled=True)
            self.puck.add_attr(rendering.Transform(translation=(500, 200)))
            self.puck.set_color(0, 0, 255)

            self.puck_trans = rendering.Transform()
            self.stick_trans = rendering.Transform()

            self.puck.add_attr(self.puck_trans)
            self.stick.add_attr(self.stick_trans)
            self.viewer.add_geom(self.goal)
            self.viewer.add_geom(self.stick)
            self.viewer.add_geom(self.puck)

            self.viewer.add_geom(self.top_wall)
            self.viewer.add_geom(self.bot_wall)

        x_pos = (self._puck_x*scale) - 100
        self.puck.set_color(0, 0, 255)
        self.puck_trans.set_translation(x_pos, self._puck_y*scale - 200)
        self.stick_trans.set_translation(8, self._robot_pos*scale - 200)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def quit_render(self):
        self.viewer.close()

    '''
    def step(self, action):
        # This will be the function used for the really world data.

        done = 0
        reward = 0
        updated = False
        hit = False
        int = False
        new_state = []
        rc = []
        self._prev_x = self._puck_x
        self._prev_y = self._puck_y

        if 0 < action < 6:      # actions 1 - 5 = Move Up
            intensity = action
            self._robot_pos += intensity
        elif 5 < action < 11:   # actions 6 - 10 = Move Down
            intensity = action - 5
            self._robot_pos -= intensity
        elif action == 11:
            self.hit_up = True
        elif action == 12:
            self.hit_down = True
        elif action == 0:       # action = Do nothing
            pass

        if 0 < action < 11:
            # Send the position of the robot to the arduino for positioning.
            # newpid = os.fork() LINUX SYSTEM CAL
            newpid = 0
            if newpid == 0:
                send(self._robot_pos)
        elif action == 11:
            # newpid = os.fork()  LINUX SYSTEM CALL
            newpid = 0
            if newpid == 0:
                send(11)  # Send whatever will be used for the hit up action
        elif action == 12:
            # newpid = os.fork() LINUX SYSTEM CALL
            newpid = 0
            if newpid == 0:
                send(12)    # Send whatever will be used for the hit up action

        while not updated:
            # Loop waits for new data to be received before moving on
            rc = recv_data()
            if rc[0] != self._prev_x and rc[1] != self._prev_y:
                updated = True

        # Update values with new data received from image processing
        self._puck_x = rc[0]
        self._puck_y = rc[1]
        self._speed_x = self.get_speed('x')
        self._speed_y = self.get_speed('y')

        # calculate reward
        x_dist = self._puck_x + 100  # checks distance between puck and dummy y axis (x = -97.5 (10/4))
        if x_dist > 100:
            x_dist = 100  # caps x_dist at 100
        y_dist = abs(self._puck_y - self._robot_pos)

        # The reward equation could be used if we implemented Joel's algorithm as an input
        if ((100 - y_dist) * (
                0.01 * (100 - x_dist))) > 50:  # Calc reward based on dist, if puck is closer, y_dist more imp
            reward += 10

        if x_dist == 0:
            # Reward for intercepting puck
            if y_dist < 5:
                reward += 100 - (y_dist * 5)
            done = True
            if y_dist < 5:
                # Check for successful hit
                if self._puck_y >= self._robot_pos and action == 12:
                    # Successful hit up action
                    reward += 1000
                    hit = True
                    if self.puck is not None:
                        self.puck.set_color(255, 0, 0)
                        time.sleep(1)
                elif self._puck_y <= self._robot_pos and action == 13:
                    # Successful hit down action
                    reward += 1000
                    hit = True
                    if self.puck is not None:
                        self.puck.set_color(255, 0, 0)
                        time.sleep(1)
        else:
            # Subtract from reward if hit action is chosen while puck is not close
            # (Will try this out later)
            reward -= 500
            pass

        new_state.append(self._robot_pos)
        new_state.append(self._puck_x)
        new_state.append(self._puck_y)
        new_state.append(self._speed_x)
        new_state.append(self._speed_y)

        return np.array(new_state), reward, int, hit, done
    '''
    def position_robot(self):
        # One strategy is to use this function and wait to call it until puck is approaching
        # Either way, robot pos will be updated dynamically while puck is moving
        move_up(self._robot_pos)
        return

    def in_play(self):
        # Return true if puck is in play
        return self._puck_x != -999 and self._puck_y != -999
    '''

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

    '''
    def step_dummy(self, action):

        done = False
        hit = False
        int = False
        reward = 0
        x_dist = 0
        y_dist = 0
        new_state = []
        self._puck_x += self._speed_x
        self._puck_y += self._speed_y

        if 0 < action < 6:      # actions 1 - 5 = Move Up
            intensity = action
            self._robot_pos += intensity
        elif 5 < action < 11:   # actions 6 - 10 = Move Down
            intensity = action - 5
            self._robot_pos -= intensity
        elif action == 11:
            self.hit_up = True
        elif action == 12:
            self.hit_down = True
        elif action == 0:       # action = Do nothing
            pass

        # stop robot at walls
        if self._robot_pos < Y_MIN:
            self._robot_pos = Y_MIN
        elif self._robot_pos > Y_MAX:
            self._robot_pos = Y_MAX

        # Stop puck
        if self._puck_x < X_MIN:
            self._puck_x = X_MIN

        # Turn puck around if it hits a wall
        if self._puck_y < Y_MIN:
            self._speed_y = -self._speed_y
            self._puck_y = Y_MIN + (Y_MIN - self._puck_y)   # Keep consistency with distance traveled
        elif self._puck_y > Y_MAX:
            self._speed_y = -self._speed_y
            self._puck_y = Y_MAX - (self._puck_y - Y_MAX)   # Keep consistency with distance traveled

        # calculate reward
        x_dist = self._puck_x + 100  # checks distance between puck and dummy y axis (x = -97.5 (10/4))
        if x_dist > 100:
            x_dist = 100  # caps x_dist at 100
        y_dist = abs(self._puck_y - self._robot_pos)

        # The reward equation could be used if we implemented Joel's algorithm as an input
        if ((100-y_dist)*(0.01*(100-x_dist))) > 50:  # Calc reward based on dist, if puck is closer, y_dist more imp
            reward += 10

        if x_dist == 0:
            # Reward for intercepting puck
            if y_dist < 5:
                int = True
                reward += 100 - (y_dist*5)
            done = True
            if y_dist < 5:
                # Check for successful hit
                if self._puck_y >= self._robot_pos and action == 11:
                    # Successful hit up action
                    reward += 1000
                    hit = True
                    if self.puck is not None:
                        self.puck.set_color(255, 0, 0)
                        time.sleep(1)
                elif self._puck_y <= self._robot_pos and action == 12:
                    # Successful hit down action
                    reward += 1000
                    hit = True
                    if self.puck is not None:
                        self.puck.set_color(255, 0, 0)
                        time.sleep(1)
        else:
            if action == 11 or action == 12:
                reward -= 100  # Teach robot not to hit unless it is time to

        new_state.append(self._robot_pos)
        new_state.append(self._puck_x)
        new_state.append(self._puck_y)
        new_state.append(self._speed_x)
        new_state.append(self._speed_y)

        return np.array(new_state), reward, int, hit, done