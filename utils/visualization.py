import sys

sys.path.append("../")

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from toy_env import vis

def visualize(intent, inits, future_time=3.5):
    """
    :param intent      - int - 0, 1, 2, 3 - integer representing our agent intent
    :param inits       - numpy array - shape(n_agent, 5) - represents initial position and velocity
    :param future_time - amount of time to run the experiment for. Defaults to 3.5
    """
    pass

def main():
    pass

if __name__ == "__main__":
    main()
