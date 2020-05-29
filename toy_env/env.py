import numpy as np

FW=0
SP=1
TL=2
TR=3

class Agent:
    # Velocity: m/s
    def __init__(self, location, direction, intention, radius, dt):
        self.location=location
        self.velocity=np.random.uniform(low=10, high=20)*direction/np.linalg.norm(direction)
        self.intention=intention
        self.dt=dt
        self.radius=radius
        
    def step(self):
        self.location=self.location+self.velocity*self.dt
        
    def execute_intention(self):
        if self.intention==FW:
            pass
        elif self.intention==SP:
            self.velocity=self.velocity*0.1**(self.dt/3)
            if np.linalg.norm(self.velocity)<0.1:
                self.velocity=self.velocity*0
        elif self.intention==TL:
            dir=self.velocity[-1::-1]*np.array([-1,1])
            dvdt=np.linalg.norm(self.velocity)**2/self.radius
            self.velocity=self.velocity+dvdt*dir/np.linalg.norm(dir)*self.dt
        elif self.intention==TR:
            dir=self.velocity[-1::-1]*np.array([1,-1])
            dvdt=np.linalg.norm(self.velocity)**2/self.radius
            self.velocity=self.velocity+dvdt*dir/np.linalg.norm(dir)*self.dt
            

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
        
        init_=[scene.starting_points[i] for i in np.random.choice(len(self.scene.starting_points), self.n)]
        for p in init_:
            self.vehicles.append(Agent(p[0], p[1], p[2], np.random.uniform(20,30),self.dt))
            
    def step(self):
        for agent in self.vehicles:
            if self.scene.check_intention(agent):
                agent.execute_intention()
            agent.step()
        return self._state()
            
    def _state(self):
        locations=[agent.location for agent in self. vehicles]
        velocities=[agent.velocity for agent in self. vehicles]
        return locations, velocities
        