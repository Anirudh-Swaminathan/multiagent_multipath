import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from env import MultiagentEnv
import math

FW=0
SP=1
TL=2
TR=3


class ExampleScene:
    def __init__(self):
        # Start location, start velocity direction, intention, goal location, goal velocity direction
        self.starting_points=[(np.array([10, -100]), np.array([0,1]), TL, np.array([-40,20]), np.array([-1,0])),
                 (np.array([20,-100]), np.array([0,1]), TR, np.array([40,40]), np.array([1,0]))]


class IntersectionScene:
    def __init__(self):
#         self.starting_points = [(np.random.uniform(0, 100, 2), np.random.uniform(0, 100, 2), np.random.choice(4)) for i
#                                 in range(6)]
        self.starting_points=[(np.array([10, -100]), np.array([0,1]), TL, np.array([-40,10]), np.array([-1,0])),
                 (np.array([20,-100]), np.array([0,1]), TR, np.array([40,30]), np.array([1,0]))]
        # bottom left coordinate and top right coordinate
        self.intersection_bounds = [(-40, -40), (40, 40)]
        self.image_bounds = [(-200, -200), (200, 200)]

    def get_scene_image(self):
        # initialize fully black image
        img = np.zeros((self.image_bounds[1][1]-self.image_bounds[0][1], self.image_bounds[1][0]-self.image_bounds[0][0]))

        # compute the coordinates of the horizontal lane
        x_axis_lower = self.intersection_bounds[0][1] - self.image_bounds[0][1]
        x_axis_higher = self.intersection_bounds[1][1] - self.image_bounds[1][1]
        img[x_axis_lower:x_axis_higher, :] = 1

        # compute the coordinates of the vertical lane
        y_axis_lower = self.intersection_bounds[0][0] - self.image_bounds[0][0]
        y_axis_higher = self.intersection_bounds[1][0] - self.image_bounds[1][0]
        img[:, y_axis_lower:y_axis_higher] = 1
        return img

    
    def dir_to_angle(self, dir_vec):
        return math.atan2(dir_vec[1], dir_vec[0])

    
    def plot_scene(self, ls, vs):
        ls = np.array(ls)
        vs = np.array(vs)
        scene_img = self.get_scene_image()
        plt.xlim(self.image_bounds[0][0], self.image_bounds[1][0])
        plt.ylim(self.image_bounds[0][1], self.image_bounds[1][1])
        plt.imshow(scene_img, cmap='gray', extent=(self.image_bounds[0][0], self.image_bounds[1][0], self.image_bounds[0][1], self.image_bounds[1][1]))
        for i in range(n):
            plt.scatter(ls[:, i, 0], ls[:, i, 1])
            d_vec = vs[-1, i, :]
            angleRad = self.dir_to_angle(d_vec)
            angle = 180.0 * angleRad / np.pi
            width = 20
            height = 10
            print(angle)
            # SE(2) -> translation + rotation for lower coordinates
            # sw = Rsb + p
            tlx = -width/2
            tly = -height/2
            lx = ls[-1, i, 0] + tlx*np.cos(angleRad) - tly*np.sin(angleRad)
            ly = ls[-1, i, 1] + tlx*np.sin(angleRad) + tly*np.cos(angleRad)
            rect = patches.Rectangle((lx, ly), width, height, angle, facecolor='red')
            plt.gca().add_patch(rect)

        plt.show()



if __name__ == '__main__':
    n = 5
    scene = IntersectionScene()
#     scene = ExampleScene()
    env = MultiagentEnv(scene, 0.001, n)
    ls = []
    vs = []
    # 4 seconds of image+trajectory; predict the next 6 seconds, probably
    disp_time = 40
    for i in range(6000):
        l, v = env.step()
        ls.append(l)
        vs.append(v)
        if i == disp_time:
            scene.plot_scene(ls, vs)
    scene.plot_scene(ls, vs)
