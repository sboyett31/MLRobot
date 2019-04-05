# Define boundaries of table
X_MAX = 100
X_MIN = -100
Y_MAX = 100
Y_MIN = 0

# Define constants for model
MAX_MEMORY = 500000         # Maximum number of samples stored at once
SAV_INCR = 99999           # How often to save memory (in # of samples updated)
BATCH = 20                  # Batch Size
NUM_EPISODES = 50000        # Total number of episodes

NUM_ENV_VAR = 5             # X, Y, X_SPEED, Y_SPEED, ROBOT_POS
NUM_ACTIONS = 13            # MOVE_UP(1-5), MOVE_DOWN(1-5), HIT_PUCK(up/down), DO NOTHING

# Define Hyper-parameters
ALPHA = 0                   # Learning rate (Can be implemented in Adam Optimizer)
GAMMA = 0.9                 # Value between 0 and 1 used for importance of future rewards (static or dynamic)
LAMBDA = .0000035           # Decay rate for epsilon  (.000005 worked pretty good for 50,000 episodes)
MAX_EPS = 0.9               # Value that epsilon starts at  (Note: Needs to be saved with memory as last eps val)
MIN_EPS = 0.05              # Value that epsilon will decay to

# File Name Constants
SAVE_MODEL = "../MLRobot/20B_neg_reward_50k.ckpt"
LOAD_MODEL = "../MLRobot/beta_save_run/nosave.ckpt"
SAVE_MEMORY = '20B_neg_reward_50k.pickle'
LOAD_MEMORY = 'beta_save_run/nosave.pickle'

