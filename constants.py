# Define boundaries of dummy table
DUMMY_X_MAX = 100
DUMMY_X_MIN = -100
DUMMY_Y_MAX = 100
DUMMY_Y_MIN = 0

# Define boundaries of real table
X_MAX = 350
X_MIN = -350
Y_MAX = 300
Y_MIN = 0

# Define constants for model
MAX_MEMORY = 500000        # Maximum number of samples stored at once
SAV_INCR = 2500             # How often to save memory (in # of samples updated)
BATCH = 20                  # Batch Size
NUM_EPISODES = 50000        # Total number of episodes

NUM_ENV_VAR = 5             # X, Y, X_SPEED, Y_SPEED, ROBOT_POS
NUM_ACTIONS = 11            # MOVE_UP(1-5), MOVE_DOWN(1-5), NO HIT_PUCK, DO NOTHING

# Define Hyper-parameters
ALPHA = 0                   # Learning rate (Can be implemented in Adam Optimizer)
GAMMA = 0.9                 # Value between 0 and 1 used for importance of future rewards (static or dynamic)
LAMBDA = .0005              # Decay rate for epsilon  (.000005 worked pretty good for 50,000 episodes)
MAX_EPS = 0.9               # Value that epsilon starts at  (Note: Needs to be saved with memory as last eps val)
MIN_EPS = 0.05              # Value that epsilon will decay to

# File Name Constants
SAVE_MODEL = "../MLRobot/first_try_save_on_pi.ckpt"
LOAD_MODEL = "noload"
SAVE_MEMORY = 'first_try_save_on_pi.pickle'
LOAD_MEMORY = 'noload'

