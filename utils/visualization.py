import sys

sys.path.append("../")

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from toy_env.vis import *
from toy_env.env_vis import *

def visualize(intents, inits, ls, vs):
    """
    :param intent - list of ints - len(intent)=n_agents - 0, 1, 2, 3 - integer representing one agent intent
    :param inits  - list - len(inits)=n_agent - represents initial position, direction, intent, goal position, goal direction for each agent in list
    :param ls     - numpy array - shape(time, n_agent, 2) - represents groundtruth trajectory
    :param vs     - numpy array - shape(time, n_agent, 2) - represents groundtruth velocities 
    """
    total_frames = 6000
    dt = 0.001
    n = len(inits)
    init_vs = vs[0, :, :]
    assert(len(intents.shape) == 1)
    assert(len(inits) == intents.shape[0])
    assert(len(inits) == init_vs.shape[0])
    old_intents = inits[:, 2]
    scene = IntersectionScene(old_intents, inits)
    f = scene.plot_scene(ls, vs)
    
    new_scene = IntersectionScene(intents, inits)
    env = MultiagentEnv(new_scene, init_vs, dt, n)
    ls = []
    vs = []
    # 4 seconds of image+trajectory; predict the next 6 seconds, probably
    total_frames = 6000
    collided = False
    for i in range(total_frames):
        c, l, v = env.step()
        collided = c or collided
        ls.append(l)
        vs.append(v)
    if collided:
        print("Collision Occurred for new intent!!")
    else:
        print("No Collisions occurred for new intent!")
    # scene.plot_scene(past_ls, past_vs)
    f_new = scene.plot_scene(ls, vs)
    return f, f_new, collided    

def main():
    s_points = np.load("../data/toydataset/2/2716/init.npy", allow_pickle=True)
    vs = np.load("../data/toydataset/2/2716/vs.npy")
    ls = np.load("../data/toydataset/2/2716/ls.npy")
    new_intents = np.array([0, 1])
    f, f_new, c = visualize(new_intents, s_points, ls, vs)
    f.savefig("test_old.png")
    f_new.savefig("test_new.png")
    f.show()
    f_new.show()
    if c:
        print("Collision occurred for new intents")
    else:
        print("Successful new trajectory was generated")

if __name__ == "__main__":
    main()
