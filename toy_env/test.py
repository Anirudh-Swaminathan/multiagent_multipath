import numpy as np
from matplotlib import pyplot as plt
from env import MultiagentEnv


class ExampleScene:
    def __init__(self):
        self.starting_points=[(np.random.uniform(0,100, 2),np.random.uniform(0,100, 2), np.random.choice(4)) for i in range(6)]
        
    def check_intention(self, agent):
        if agent.location[0]>-20 and agent.location[0]<70 and agent.location[1]>-20 and agent.location[1]<70:
            return np.random.choice([0,1],p=[0.2, 0.8])
        
        
if __name__=='__main__':
    n=5
    scene=ExampleScene()
    env=MultiagentEnv(scene,0.1,n)
    ls=[]
    vs=[]
    for i in range(100):
        l,v=env.step()
        ls.append(l)
        vs.append(v)
    ls=np.array(ls)
    vs=np.array(vs)
    for i in range(n):
        plt.scatter(ls[:,i,0],ls[:,i,1])
    plt.show()
    