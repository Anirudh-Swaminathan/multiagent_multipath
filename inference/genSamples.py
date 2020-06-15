import sys

sys.path.append("../")

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from toy_env.vis import *
from toy_env.env_vis import *

FW=0
SP=1
TL=2
TR=3

def visualize(intents, inits, init_vs, filePath):
    """
    :param intent - list of ints - len(intent)=n_agents - 0, 1, 2, 3 - integer representing one agent intent
    :param inits  - list - len(inits)=n_agent - represents initial position, direction, intent, goal position, goal direction for each agent in list 
    """
    
    total_frames = 6000
    dt = 0.001
    n = len(inits)
    
    assert(len(intents.shape) == 1)
    assert(len(inits) == intents.shape[0])
        
    new_scene = IntersectionScene(intents, inits)
    env = MultiagentEnv(new_scene, init_vs, dt, n)
    
    past_ls = []
    past_vs = []
    
    ls = []
    vs = []
    total_frames = 6000
    collided = False
    folder_name = 'ExampleScenes/{}/'.format(filePath)
    try:
        os.mkdir(folder_name)
    except:
        pass
    
    for i in range(total_frames):
        c, l, v = env.step()
        collided = c or collided
        ls.append(l)
        vs.append(v)
        if i == 2500:
            f_new = new_scene.plot_scene(ls, vs)
            f_new.savefig(folder_name + "scene.png")
    
    inits = np.array(inits)
    ls = np.array(ls)
    vs = np.array(vs)
    
    f_new = new_scene.plot_scene(ls, vs)
    f_new.savefig(folder_name + "full_scene.png")
        
    np.save(folder_name + "init.npy", inits)
    np.save(folder_name + "ls.npy", ls)
    np.save(folder_name + "vs.npy", vs)
    if collided:
        print("Collided")
    
    return

def main():
    n_agents = 2
    starting_points = list() 
    intents = np.array([TL,TR])
    init_vs = []
    
    #starting_points = list of 5 elements
    #1. np array: initial location of each object of the number of agents x 2
    #2. direction of each agent as num_objects x 2
    #3. intent of each agent as num_agents x 1
    #4. goal position of each agent as num_agents x 2
    
    for agent in range(n_agents):
        if agent == 0:
            agent_pos = np.array([0, -100])
            agent_dir = np.array([0, 1])
            agent_intent = intents[agent]
            agent_goal_pos = np.array([-50,10])
            agent_goal_dir = np.array([-1, 0])
            agent_init_vs = np.random.uniform(low=25, high=35)*agent_dir/np.linalg.norm(agent_dir)
        elif agent == 1:
            agent_pos = np.array([-100, 0])
            agent_dir = np.array([1, 0])
            agent_intent = intents[agent]
            agent_goal_pos = np.array([10,-50])
            agent_goal_dir = np.array([0, -1])
            agent_init_vs = np.random.uniform(low=25, high=35)*agent_dir/np.linalg.norm(agent_dir)
            
        agent_sp = (agent_pos, agent_dir, agent_intent, agent_goal_pos, agent_goal_dir)
        starting_points.append(agent_sp)
        
        init_vs.append(agent_init_vs)

    intents = np.array(intents)    
    starting_points = np.array(list(np.array(starting_points)))
    init_vs = np.array(init_vs)
    
    visualize(intents, starting_points, init_vs, 3)
    
if __name__ == "__main__":
    main()
