import numpy as np
from scene import *
from env import *
import os
from multiprocessing import Process

def generate_sample(id, n):
    path = "../data/dataset_vary/"+str(n)+'/'
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
            plts[0].savefig(folder+'scene_1.0.png', dpi='figure')
            plts[1].savefig(folder+'scene_1.5.png', dpi='figure')
            plts[2].savefig(folder+'scene_2.0.png', dpi='figure')
            plts[3].savefig(folder+'scene.png', dpi='figure')
            print(plt.get_fignums())
            
            print('Collisions:', collision_count)
        for f in plt.get_fignums():
                plt.close(f)

def generate_wrapper(n):
    os.mkdir("../data/dataset_vary/"+str(n)+'/')
    for i in range(1000):
        generate_sample(i, n)
        print(n, i)



if __name__ == '__main__':
    p=[]
    for n in [4,3,2,5]:
        p=Process(target=generate_wrapper, args=(n,))
        p.start()
            
    