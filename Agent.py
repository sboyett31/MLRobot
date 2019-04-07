import tensorflow as tf
import numpy as np
import math
import random as r
import pickle as p
import statistics as s
import time
import gym
import os
#import matplotlib.pyplot as plt
from AirHockeyEnv import AirHockeyEnv
from constants import NUM_ENV_VAR, NUM_ACTIONS, ALPHA, GAMMA, LAMBDA, \
    MAX_EPS, MIN_EPS, MAX_MEMORY, BATCH, NUM_EPISODES, SAV_INCR, SAVE_MODEL, \
    LOAD_MODEL, SAVE_MEMORY, LOAD_MEMORY


render_flag = False

class Model:
    def __init__(self, num_states, num_actions, batch_size):
        self._num_states = num_states
        self._num_actions = num_actions
        self._batch_size = batch_size
        # define the placeholders
        self._states = None
        self._actions = None
        # the output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None
        # now setup the model
        self._define_model()

    def _define_model(self):
        self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
        # create two fully connected hidden layers using activation function ReLU
        fc1 = tf.layers.dense(self._states, 50, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 50, activation=tf.nn.relu)
        # fc3 = tf.layers.dense(fc2, 50, activation=tf.nn.relu)   # Added third layer to try it out
        self._logits = tf.layers.dense(fc2, self._num_actions)  # defaults to linear activation function
        # defines the type of loss we are using (mean squared error loss)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)      # Figure out how to get value of loss
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)           # Learning rate parameter(default .001)
        self._var_init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def predict_one(self, state, sess):
        # returns the output of the network with an input of a single state
        # by calling the _logits operation
        # (reshaping method is used to ensure data has size of (1, num_states)
        return sess.run(self._logits, feed_dict={self._states:
                        state.reshape(1, self._num_states)})

    def predict_batch(self, states, sess):
        # Predicts a whole batch of outputs when given multiple input states
        # This is used to perform batch evaluation of Q(s,a) and Q(s',a') values for training.
        return sess.run(self._logits, feed_dict={self._states: states})

    def train_batch(self, sess, x_batch, y_batch):
        # Takes a batch training step of the network
        sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})


class Memory:
    #  This class stores all of the results of the actions taken by the agent during the game.
    #  This class also handles the retrieval of those results which can be used to batch
    #  train the network.

    def __init__(self, max_memory):
        # max_memory argument controls the maximum number of (state,action,reward,next_state)
        # tuples that the _samples list can hold.
        # (The bigger the better) -- this ensures better random mixing of the samples.
        # Make sure not to run into memory errors
        self._max_memory = max_memory
        self._samples = []
        self._sample_count = 0
        # Memory Size Indicator
        self.full = False
        # Save memory between runs
        self._file_full = False
        self._save_count = 0
        self._datafile = None
        # Rad memory at start up
        self.read_mem()

    def add_sample(self, sample):
        # Takes an individual tuple and appends it to the _samples list
        # Pops in FIFO manner if max_memory is reached
        # print(sample)
        self._samples.append(sample)
        self._save_count += 1
        if len(self._samples) > self._max_memory:
            if not self.full:
                print("**********MEMORY FULL**********")
                self.full = True
            self._samples.pop(0)  # FIFO
        if self._save_count > SAV_INCR:
            self.save_mem()
            self._save_count = 0

    def sample(self, no_samples):
        # returns a randomly selected sample
        # if no_samples is larger than the actual memory, whatever is available in
        # the memory is returned
        if no_samples > len(self._samples):
            return r.sample(self._samples, len(self._samples))
        else:
            return r.sample(self._samples, no_samples)

    def save_mem(self):
        # Linux Fork Operation commented out
        # newpid = os.fork()
        newpid = 0
        if newpid == 0:
            print("Writing memory...")
            save_samples = self._samples
            save_file = open(SAVE_MEMORY, 'wb')
            p.dump(save_samples, save_file)
            save_file.close()
            print("Complete")
        else:
            pass

    def read_mem(self):
        # Read memory from file at startup using deserialization
        try:
            print("Checking for memory file.")
            self._datafile = open(LOAD_MEMORY, 'rb')
            print("Reading memory...")
            self._samples = p.load(self._datafile)
            print("Complete")
        except FileNotFoundError:
            # No Memory to Read
            print("No memory to read... Starting with empty memory.")


class GameRunner:
    # Where model dynamics, agent action, and training is organized.
    def __init__(self, tf_sess, model, env, memory, max_eps, min_eps, decay, render=False):
        self._sess = tf_sess        # TensorFlow session object
        self._env = env             # Open AI gym environment
        self._model = model         # Neural Network Model
        self._memory = memory       # Instance of Memory Class
        self._render = render       # boolean which determines if game env is shown on screen
        self._max_eps = max_eps     # Max epsilon value (starting value)
        self._min_eps = min_eps     # Min epsilon value (value to which it will decay)
        self._decay = decay         # Rate at which epsilon will decay
        self._eps = self._max_eps   # Initialize epsilon to max value
        self._steps = 0
        self._reward_list = []
        self.hit = 0
        self.int = 0

    def run(self):
        # This is the execution of one "Episode"
        # Episode - One sequence of states, actions, and rewards that ends in a terminal state
        # State: Rpos, Xpos, Ypos, Xspeed, Yspeed
        state = self._env.reset()           # Resetting the environment
        tot_reward = 0                      # Setting tot_reward = 0
        self.hit = 0

        while True:
            if self._render or render_flag:
                self._env.render()
            # Initialize action to 0 so that no action will be taken unless puck is on our side
            action = 0
            if state[1] < 0:
                # Only react if puck is on our side of env
                action = self._choose_action(state)
                print("action is: {}".format(action))


            next_state, reward, self.int, self.hit, done = self._env.step(action)

            '''
            Here we can move the robot to the position to hit the puck if the puck is close enough based on delay
            
            '''

            if state[1] < 0:
                # Neural Net ignores states where puck is not on our side
                tot_reward += reward
                if done:
                    next_state = None
                self._memory.add_sample((state, action, reward, next_state))
                self._replay()
                # decay epsilon value
                self._steps += 1
                self._eps = self._min_eps + (self._max_eps - self._min_eps) \
                    * math.exp(-self._decay * self._steps)
            # update state
            state = next_state
            # if the episode is done, break the loop
            if done:
                self._reward_list.append(tot_reward)
                break

    def _choose_action(self, state):
        if r.random() < self._eps:
            return r.randint(0, self._model._num_actions - 1)
        else:
            return np.argmax(self._model.predict_one(state, self._sess))

    def _replay(self):
        batch = self._memory.sample(self._model._batch_size)
        states = np.array([val[0] for val in batch])
        next_states = np.array([(np.zeros(self._model._num_states)
                                 if val[3] is None else val[3]) for val in batch])
        # predict Q(s,a) given the batch of states
        q_s_a = self._model.predict_batch(states, self._sess)
        # predict Q(s', a') - so that we can do gamma * max(Q(s', a')) below
        q_s_a_d = self._model.predict_batch(next_states, self._sess)
        # setup training arrays
        x = np.zeros((len(batch), self._model._num_states))
        y = np.zeros((len(batch), self._model._num_actions))
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            # get the current q values for all actions in state
            current_q = q_s_a[i]
            # update the q value for action
            if next_state is None:
                # The episode is completed, so there is no next state
                current_q[action] = reward
            else:
                current_q[action] = reward + GAMMA * np.amax(q_s_a_d[i])
            x[i] = state
            y[i] = current_q
        self._model.train_batch(self._sess, x, y)


def save_model(saver, ss):
    # Function in order to implement saving using multi threading
    # newpid = os.fork()  LINUX OS CALL
    newpid = 0
    if newpid == 0:
        save_path = saver.save(ss, SAVE_MODEL)
        print("Model saved in path %s" % save_path)
    else:
        pass


def save_eps(eps):
    #  Save epsilon value between runs
    save_file = open('save_eps.txt', 'w')
    save_file.write(str(eps))
    save_file.close()


if __name__ == "__main__":

    env = AirHockeyEnv()
    model = Model(NUM_ENV_VAR, NUM_ACTIONS, batch_size=BATCH)
    mem = Memory(MAX_MEMORY)

    with tf.Session() as sess:
        # sess.run(model._var_init)
        try:
            model.saver.restore(sess, LOAD_MODEL)
            print("Model loaded!")
        except:
            # If there is no prior model to restore
            print("No previous model to load...")
            print("Initializing Artificial Neural Network Model")
            sess.run(model._var_init)

        gr = GameRunner(sess, model, env, mem, MAX_EPS, MIN_EPS, LAMBDA)
        num_episodes = NUM_EPISODES
        count = 0
        episode_hits = 0
        episode_ints = 0
        li = []
        mem_full_reward = []
        int_pct_arr = []
        hit_pct_arr = []
        while count < num_episodes:
            if gr.int:
                episode_ints += 1
            if gr.hit:
                episode_hits += 1
            if count % 10 == 0:
                if count != 0 and count % 100 == 0:
                    # Print Stats every 100 episodes
                    li = gr._reward_list[count-100:count-1]
                    avg_rwd = sum(li) / float(len(li))
                    int_pct = episode_ints
                    hit_pct = episode_hits
                    int_pct_arr.append(int_pct)
                    hit_pct_arr.append(hit_pct)
                    print('Episode {} of {}.  Eps Value: {:.4f}, Avg Reward: {:.2f}, Int Rate: {}% Hit Rate: {}%'
                          .format(count+1, num_episodes, gr._eps, avg_rwd, int_pct, hit_pct))
                    episode_ints = 0
                    episode_hits = 0

            elif count % 999 == 0:
                # Saving Model and Epsilon Value every 1000 Episodes
                save_sess = sess
                save_model(model.saver, save_sess)
                save_eps(gr._eps)
            gr.run()
            count += 1
            if count/num_episodes > .99 and not render_flag:
                # Display Simulation after 99% Of episodes have been complete
                render_flag = True
        # Plot Results
        #plt.plot(gr._reward_list, 'b')  # mem_full_reward, 'r')
        #plt.show()
        #plt.plot(int_pct_arr, 'k')
        #plt.show()
        #plt.plot(hit_pct_arr, 'g')
        #plt.show()
        # plt.close("all")
        # plt.plot(gr.max_x_store)
        # plt.show()
