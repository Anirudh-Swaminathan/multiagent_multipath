import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import math

FW=0
SP=1
TL=2
TR=3

LANE_BOT = np.array([0, 1])
LANE_TOP = np.array([0, -1])
LANE_LEFT = np.array([1, 0])
LANE_RIGHT = np.array([-1, 0])


class IntersectionScene:
    def __init__(self, intents, starting_points):
        assert(len(intents) == len(starting_points))
        # bottom left coordinate and top right coordinate
        self.intersection_bounds = [(-40, -40), (40, 40)]
        self.image_bounds = [(-200, -200), (200, 200)]

        # maximum distance away from intersection to generate the vehicle
        self.closeness = 75
        self.vw = 10
        self.vh = 20
        # distance from intersection minimum
        self.iw = 60
        self.ih = 60
        
        self.starting_points = list()
        for i in range(len(intents)):
            sp = np.copy(starting_points[i])
            sp[2] = intents[i]
            if np.allclose(sp[1], LANE_LEFT):
                self.gen_left(sp)
            elif np.allclose(sp[1], LANE_RIGHT):
                self.gen_right(sp)
            elif np.allclose(sp[1], LANE_TOP):
                self.gen_top(sp)
            elif np.allclose(sp[1], LANE_BOT):
                self.gen_bot(sp)
                
    def gen_bot(self, sp):
        # bottom lane => x in (intersection_bounds+vehicle_width/2, intersection_bounds-vehicle_width/2); y in (-(40+vehicle_length/2) to -(40 + closeness))
        sbot_posx = sp[0][0]
        sbot_posy = sp[0][1]
        sbot_pos = np.array([sbot_posx, sbot_posy])
        sbot_vel = sp[1]
        sbot_int = sp[2]
        # compute goal positions and directions
        if sbot_int == FW or sbot_int == SP:
            sbot_gol = np.array([sbot_posx, sbot_posy+self.closeness])
            sbot_gvel = np.copy(sbot_vel)
        elif sbot_int == TL:
            sbot_gvel = np.array([-1, 0])
            sbot_golx = np.random.uniform(self.intersection_bounds[0][0]-self.ih/2, self.intersection_bounds[0][0]-self.closeness)
            sbot_goly = np.random.uniform(self.intersection_bounds[0][1]+self.vw/2, self.intersection_bounds[1][1]-self.vw/2)
            sbot_gol = np.array([sbot_golx, sbot_goly])
        elif sbot_int == TR:
            sbot_gvel = np.array([1, 0])
            sbot_golx = np.random.uniform(self.intersection_bounds[1][0]+self.ih/2, self.intersection_bounds[1][0]+self.closeness)
            sbot_goly = np.random.uniform(self.intersection_bounds[0][1]+self.vw/2, self.intersection_bounds[1][1]-self.vw/2)
            sbot_gol = np.array([sbot_golx, sbot_goly])
        # append the entire state to a list
        sbot = tuple((sbot_pos, sbot_vel, sbot_int, sbot_gol, sbot_gvel))
        self.starting_points.append(sbot)   
        
    def gen_right(self, sp):
        # right lane => x in (intersection_bounds+vehicle_width/2, intersection_bounds+closeness); y in (-(40+vehicle_width/2), (40 + vehcile_width/2))
        sr_posx = sp[0][0]
        sr_posy = sp[0][1]
        sr_pos = np.array([sr_posx, sr_posy])
        # velocity direction
        sr_vel = sp[1]
        # intention
        sr_int = sp[2]
        # compute the goal positions and velocities
        if sr_int == FW or sr_int == SP:
            sr_gvel = np.copy(sr_vel)
            sr_gol = np.array([sr_posx-self.closeness, sr_posy])
        elif sr_int == TL:
            sr_gvel = np.array([0, -1])
            sr_golx = np.random.uniform(self.intersection_bounds[0][0]+self.vw/2, self.intersection_bounds[1][0]-self.vw/2)
            sr_goly = np.random.uniform(self.intersection_bounds[0][1]-self.ih/2, self.intersection_bounds[0][1]-self.closeness)
            sr_gol = np.array([sr_golx, sr_goly])
        elif sr_int == TR:
            sr_gvel = np.array([0, 1])
            sr_golx = np.random.uniform(self.intersection_bounds[0][0]+self.vw/2, self.intersection_bounds[1][0]-self.vw/2)
            sr_goly = np.random.uniform(self.intersection_bounds[1][1]+self.ih/2, self.intersection_bounds[1][1]+self.closeness)
            sr_gol = np.array([sr_golx, sr_goly])
        sr = tuple((sr_pos, sr_vel, sr_int, sr_gol, sr_gvel))
        self.starting_points.append(sr)
    
    def gen_top(self, sp):
        # top lane => x in (intersection_bounds+vehicle_width/2, intersection_bounds-vehicle_width/2); y in ((40+vehicle_length/2), (40 + closeness))
        stop_posx = 
        stop_pos = np.array([stop_posx, stop_posy])
        stop_vel = sp[1]
        # random choice of turn
        stop_int = sp[2]
        # compute goal position and direction
        if stop_int == FW or stop_int == SP:
            stop_gvel = np.copy(stop_vel)
            stop_gol = np.array([stop_posx, stop_posy-self.closeness]) 
        elif stop_int == TL:
            stop_gvel = np.array([1, 0])
            stop_golx = np.random.uniform(self.intersection_bounds[1][0]+self.ih/2, self.intersection_bounds[1][0]+self.closeness)
            stop_goly = np.random.uniform(self.intersection_bounds[0][1]+self.vw/2, self.intersection_bounds[1][1]-self.vw/2)
            stop_gol = np.array([stop_golx, stop_goly])
        elif stop_int == TR:
            stop_gvel = np.array([-1, 0])
            stop_golx = np.random.uniform(self.intersection_bounds[0][0]-self.ih/2, self.intersection_bounds[0][0]-self.closeness)
            stop_goly = np.random.uniform(self.intersection_bounds[0][1]+self.vw/2, self.intersection_bounds[1][1]-self.vw/2)
            stop_gol = np.array([stop_golx, stop_goly])
        stop = tuple((stop_pos, stop_vel, stop_int, stop_gol, stop_gvel))
        self.starting_points.append(stop)
        
    def gen_left(self, sp):
        # left lane => x in (-intersection_bounds-vehicle_width/2, -intersection_bounds-closeness); y in (-(40+vehicle_width/2), (40 + vehcile_width/2))
        sl_posx = sp[0][0]
        sl_posy = sp[0][1]
        sl_pos = np.array([sl_posx, sl_posy])
        # velocity direction
        sp_vel = sp[1]
        # random intention
        sl_int = sp[2]
        # compute the goal positions and velocities
        if sl_int == FW or sl_int == SP:
            sl_gvel = np.copy(sl_vel)
            sl_gol = np.array([sl_posx + self.closeness, sl_posy])
        elif sl_int == TL:
            sl_gvel = np.array([0, 1])
            sl_golx = np.random.uniform(self.intersection_bounds[0][0]+self.vw/2, self.intersection_bounds[1][0]-self.vw/2)
            sl_goly = np.random.uniform(self.intersection_bounds[1][1]+self.ih/2, self.intersection_bounds[1][1]+self.closeness)
            sl_gol = np.array([sl_golx, sl_goly])
        elif sl_int == TR:
            sl_gvel = np.array([0, -1])
            sl_golx = np.random.uniform(self.intersection_bounds[0][0]+self.vw/2, self.intersection_bounds[1][0]-self.vw/2)
            sl_goly = np.random.uniform(self.intersection_bounds[0][1]-self.ih/2, self.intersection_bounds[0][1]-self.closeness)
            sl_gol = np.array([sl_golx, sl_goly])  
        sl = tuple((sl_pos, sl_vel, sl_int, sl_gol, sl_gvel))
        self.starting_points.append(sl)

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
    from env import MultiagentEnv
    n = 2
    scene = IntersectionScene(intents, s_points)
    env = MultiagentEnv(scene, 0.001, n)
    ls = []
    vs = []
    past_ls = []
    past_vs = []
    future_ls = []
    future_vs = []
    # 4 seconds of image+trajectory; predict the next 6 seconds, probably
    total_frames = 6000
    disp_time = np.rint(0.4*total_frames)
    collided = False
    for i in range(total_frames):
        c, l, v = env.step()
        collided = c or collided
        ls.append(l)
        vs.append(v)
        if i <= disp_time:
            past_ls.append(l)
            past_vs.append(v)
        else:
            future_ls.append(l)
            future_vs.append(v)
        if i == disp_time:
            scene.plot_scene(past_ls, past_vs)
    if collided:
        print("Collision Occurred!!")
    else:
        print("No Collisions occurred!")
        # this plot can be changed to save. Only display/save if it did not have any collisions
        # also save the past + future trajectories
        # TODO save image + past + future trajectories
        scene.plot_scene(past_ls, past_vs)
        scene.plot_scene(ls, vs)
        future_ls = np.array(future_ls)
        future_vs = np.array(future_vs)
        print(future_ls.shape)
        print(future_vs.shape)

