import numpy as np
from matplotlib import pyplot as plt
from env import MultiagentEnv


class ExampleScene:
    def __init__(self):
        self.starting_points = [(np.random.uniform(0, 100, 2), np.random.uniform(0, 100, 2), np.random.choice(4)) for i
                                in range(6)]

    def check_intention(self, agent):
        if agent.location[0] > -20 and agent.location[0] < 70 and agent.location[1] > -20 and agent.location[1] < 70:
            return np.random.choice([0, 1], p=[0.2, 0.8])


class IntersectionScene:
    def __init__(self):
        self.starting_points = [(np.random.uniform(0, 100, 2), np.random.uniform(0, 100, 2), np.random.choice(4)) for i
                                in range(6)]
        # bottom left coordinate and top right coordinate
        self.intersection_bounds = [(-40, -40), (40, 40)]
        self.image_bounds = [(-200, -200), (200, 200)]

    def check_intention(self, agent):
        if agent.location[0] > -20 and agent.location[0] < 70 and agent.location[1] > -20 and agent.location[1] < 70:
            return np.random.choice([0, 1], p=[0.2, 0.8])

    def get_scene_image(self):
        # initialize fully black image
        img = np.zeros((self.image_bounds[1][1]-self.image_bounds[0][1], self.image_bounds[1][0]-self.image_bounds[0][0]))
        print(img.shape)
        #plt.imshow(img, cmap='gray')
        #plt.show()

        # compute the coordinates of the horizontal lane
        x_axis_lower = self.intersection_bounds[0][1] - self.image_bounds[0][1]
        x_axis_higher = self.intersection_bounds[1][1] - self.image_bounds[1][1]
        img[x_axis_lower:x_axis_higher, :] = 1
        #plt.imshow(img, cmap='gray')
        #plt.show()

        # compute the coordinates of the vertical lane
        y_axis_lower = self.intersection_bounds[0][0] - self.image_bounds[0][0]
        y_axis_higher = self.intersection_bounds[1][0] - self.image_bounds[1][0]
        img[:, y_axis_lower:y_axis_higher] = 1
        #plt.imshow(img, cmap='gray')
        #plt.show()
        return img

    
    def dir_to_angle(self, dir_vec):
        dvec = dir_vec / np,linalg.norm(dir_vec)
        return np.atan2(dvec[1], dvec[0])

    
    def plot_scene(self, ls, vs):
        ls = np.array(ls)
        vs = np.array(vs)
        scene_img = self.get_scene_image()
        print(scene_img.shape, scene_img.dtype)
        plt.xlim(self.image_bounds[0][0], self.image_bounds[1][0])
        plt.ylim(self.image_bounds[0][1], self.image_bounds[1][1])
        plt.imshow(scene_img, cmap='gray', extent=(self.image_bounds[0][0], self.image_bounds[1][0], self.image_bounds[0][1], self.image_bounds[1][1]))
        for i in range(n):
            plt.scatter(ls[:, i, 0], ls[:, i, 1])
        plt.show()



if __name__ == '__main__':
    n = 2
    scene = IntersectionScene()
    env = MultiagentEnv(scene, 0.1, n)
    ls = []
    vs = []
    for i in range(100):
        l, v = env.step()
        ls.append(l)
        vs.append(v)
    scene.plot_scene(ls, vs)
