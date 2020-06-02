import numpy as np

FW=0
SP=1
TL=2
TR=3

class Agent:
    # Velocity: m/s
    def __init__(self, location, direction, intention, dt):
        self.location=location
        self.velocity=np.random.uniform(low=20, high=60)*direction/np.linalg.norm(direction)
        self.intention=intention
        self.brake_flag=False
        self.dt=dt
        
    def step(self):
        self.location=self.location+self.velocity*self.dt
        
    def set_goal(self, goal_loc, goal_vel):
        self.goal_loc=goal_loc
        self.goal_vel=goal_vel/np.linalg.norm(goal_vel)
        
    def turn_check(self):
        sdir=self.velocity[-1::-1]*np.array([-1,1])
        sdir=sdir/np.linalg.norm(sdir)
        gdir=self.goal_vel[-1::-1]*np.array([-1,1])
        d=self.location-self.goal_loc
        theta1=np.arccos(np.abs(d@gdir/np.linalg.norm(d)))
        theta2=np.arccos(np.abs(d@sdir/np.linalg.norm(d)))
        if np.abs(np.pi/2-theta1)<1e-2:
            return None
        if np.abs(theta2-theta1)<1e-2:
            return np.linalg.norm(d)/2/np.cos(theta1)
        elif theta1-theta2>1e-2:
            return 30
        
    def execute_intention(self):
        if self.intention==FW:
            pass
        elif self.intention==SP:
            if np.linalg.norm(self.location-self.goal_loc)<1e-1:
                self.brake_flag=True
            if self.brake_flag:
                self.velocity=self.velocity*0.1**(self.dt/3)
                if np.linalg.norm(self.velocity)<0.1:
                    self.velocity=self.velocity*0
        elif self.intention==TL:
            r=self.turn_check()
            if r is not None:
                dir=self.velocity[-1::-1]*np.array([-1,1])
                dvdt=np.linalg.norm(self.velocity)**2/r
                self.velocity=self.velocity+dvdt*dir/np.linalg.norm(dir)*self.dt
                if np.arccos(self.velocity@self.goal_vel/np.linalg.norm(self.velocity))<1e-2:
                    self.velocity=np.linalg.norm(self.velocity)*self.goal_vel
                    self.intention=FW
        elif self.intention==TR:
            r=self.turn_check()
            if r is not None:
                dir=self.velocity[-1::-1]*np.array([1,-1])
                dvdt=np.linalg.norm(self.velocity)**2/r
                self.velocity=self.velocity+dvdt*dir/np.linalg.norm(dir)*self.dt
                if np.arccos(self.velocity@self.goal_vel/np.linalg.norm(self.velocity))<1e-2:
                    self.velocity=np.linalg.norm(self.velocity)*self.goal_vel
                    self.intention=FW
            

class MultiagentEnv:
    # scene: Binary mask of the map
        # scene.starting_points: ((x,y),(direction_x, direction_y), intention) pair. Vehicle's velocity is the same as direction when initialized
        # scene.check_intention(agent): returns whether the intention of the agent should be executed
    # dt: Simulation timestep
    # n: Number of vehicles
    
    def __init__(self, scene, dt, n=2):
        self.scene=scene
        self.dt=dt
        self.n=n
        self.vehicles=[]
        
        #init_=[scene.starting_points[i] for i in np.random.choice(len(self.scene.starting_points), self.n)]
        init_ = [scene.starting_points[i] for i in np.random.permutation(len(self.scene.starting_points))[:self.n]]
        for p in init_:
            a=Agent(p[0], p[1], p[2], self.dt)
            a.set_goal(p[3], p[4])
            self.vehicles.append(a)
            
    def step(self):
        for agent in self.vehicles:
            agent.execute_intention()
            agent.step()
        return self._state()
            
    def _state(self):
        locations=[agent.location for agent in self. vehicles]
        velocities=[agent.velocity for agent in self. vehicles]
        return locations, velocities
