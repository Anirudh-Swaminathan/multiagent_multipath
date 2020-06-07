import numpy as np
from scene import *
from env import *
import os

path = "./dataset/"

def generate_sample(id):
    n = 2
    # 4 seconds of image+trajectory; predict the next 6 seconds, probably
    total_frames = 6000
    disp_time = 2500
    success = False
    while not success:
        scene = IntersectionScene()
        env = MultiagentEnv(scene, 0.001, n)
        starting_points = env.init_
        ls = []
        vs = []
        for i in range(total_frames):
            collided, l, v = env.step()
            if collided:
                break
            ls.append(l)
            vs.append(v)
            if i == disp_time:
                plt=scene.plot_scene(ls, vs)
        if i!=total_frames-1:
            continue
        folder=path+str(id)+'/'
        try:
            os.mkdir(folder)
        except:
            pass
        np.save(folder+'ls', ls)
        np.save(folder+'vs', vs)
        np.save(folder+'init', starting_points)
        plt.savefig(folder+'scene', dpi='figure')
        success=True



if __name__ == '__main__':
    #generate_sample(1)
    for i in range(5000):
        generate_sample(i)
        print(i)