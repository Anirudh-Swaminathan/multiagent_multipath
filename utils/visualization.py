import sys

sys.path.append("../")

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from toy_env import vis

def visualize(intent, inits, init_vs, future_time=3.5):
    """
    :param intent      - int - 0, 1, 2, 3 - integer representing our agent intent
    :param inits       - numpy array - shape(n_agent, 5) - represents initial position, direction, intent, goal position, goal direction
    :param init_vs     - numpy array - shape(n_agent, )  - represents initial velocities
    :param future_time - amount of time to run the experiment for. Defaults to 3.5
    """
    pass

def main():
    pass

if __name__ == "__main__":
    main()
