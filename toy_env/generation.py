import numpy as np
from scene import *
from env import *
import os

path = "./dataset_largecollision/"

def generate_sample(id):
    n = 2
    # 4 seconds of image+trajectory; predict the next 6 seconds, probably
    total_frames = 6000
    disp_time = [1000, 1500, 2000, 2500]
    success = False
    collision_count=0
    while not success:
        plts=[]
        scene = IntersectionScene()
        env = MultiagentEnv(scene, 0.001, n)
        starting_points = env.init_
        ls = []
        vs = []
        for i in range(total_frames):
            collided, l, v = env.step()
            if collided:
                collision_count=collision_count+1
                break
            ls.append(l)
            vs.append(v)
            if i in disp_time:
                plts.append(scene.plot_scene(ls, vs))
        if i==total_frames-1:
            success=True
            
        if success:
            folder=path+str(id)+'/'
            try:
                os.mkdir(folder)
            except:
                pass
            np.save(folder+'ls', ls)
            np.save(folder+'vs', vs)
            np.save(folder+'init', starting_points)
            plts[0].savefig(folder+'scene.png', dpi='figure')
            plts[1].savefig(folder+'scene_1.5.png', dpi='figure')
            plts[2].savefig(folder+'scene_2.0.png', dpi='figure')
            plts[3].savefig(folder+'scene_2.5.png', dpi='figure')
            print(plt.get_fignums())
            for f in plt.get_fignums():
                plt.close(f)
        
            print('Collisions:', collision_count)



if __name__ == '__main__':
    #generate_sample(1)
    for i in range(5000):
        generate_sample(i)
        print(i)