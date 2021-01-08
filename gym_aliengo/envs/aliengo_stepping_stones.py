import gym
from gym import error, spaces, utils
from gym.utils import seeding
import sys
import pybullet as p
import os
import time
import numpy as np
import warnings
from cv2 import putText, FONT_HERSHEY_SIMPLEX, imwrite, cvtColor, COLOR_RGB2BGR
from gym_aliengo.envs import aliengo
from pybullet_utils import bullet_client as bc

'''
TODO
- log distance traveled
- use createCollisionShapeArray
'''
class AliengoSteppingStones(gym.Env):
    def __init__(self, render=False, realTime=False):
        # Environment Options
        self._apply_perturbations = False
        self.perturbation_rate = 0.00 # probability that a random perturbation is applied to the torso
        self.max_torque = 40.0 
        self.kp = 1.0 
        self.kd = 1.0
        self.n_hold_frames = 1 
        self._is_render = render
        self.eps_timeout = 240.0/self.n_hold_frames * 20 # number of steps to timeout after
        self.realTime = realTime

        # stepping stone parameters
        self.height = 1.0 # height of the heightfield
        self.course_length = 5.0 # total distance from edge of start block to edge of end block 
        self.course_width = 2.0 # widght of path of stepping stones 
        self.stone_length = 0.25 # side length of square stepping stones
        self.stone_density = 6.0 # stones per square meter 
        self.stone_height_range = 0.25 # heights of stones will be within [self.height - this/2, self.height + this/2 ]
        self.ray_start_height = self.height + self.stone_height_range

        # heightmap param dict
        self.heightmap_params = {'length': 1.25, # assumes square 
                            'robot_position': 0.5, # distance of robot base origin from back edge of height map
                            'grid_spacing': 0.125}
        assert self.heightmap_params['length'] % self.heightmap_params['grid_spacing'] == 0
        self.grid_len = int(self.heightmap_params['length']/self.heightmap_params['grid_spacing']) + 1

        if self._is_render:
            self.client = bc.BulletClient(connection_mode=p.GUI)
        else:
            self.client = bc.BulletClient(connection_mode=p.DIRECT)
        self.fake_client = bc.BulletClient(connection_mode=p.DIRECT) # this is only used for getting the heightmap 


        if (self.client == -1) or (self.fake_client == -1):
            raise RuntimeError('Pybullet could not connect to physics client')


    
        # (50) applied torque, pos, and vel for each motor, base orientation (quaternions), foot normal forces,
        # cartesian base acceleration, base angular velocity
        self.state_space_dim = 12 * 3 + 4 + 4 + 3 + 3 + self.grid_len**2
        self.num_joints = 18 # This includes fixed joints from the URDF
        self.action_space_dim = 12 # this remains unchanged

        self.state = np.zeros(self.state_space_dim) # I currently don't distinguish between state and observation
        self.applied_torques = np.zeros(12) 
        self.joint_velocities = np.zeros(12)
        self.joint_positions = np.zeros(12)
        self.base_orientation = np.zeros(4)
        self.foot_normal_forces = np.zeros(4)
        self.cartesian_base_accel = np.zeros(3) 
        self.base_twist = np.zeros(6) # used to calculate accelerations, angular vel included in state
        self.previous_base_twist = np.zeros(6) # used to calculate accelerations, angular vel included in state
        self.base_position = np.zeros(3) # not returned as observation, but used for calculating reward or termination
        self.eps_step_counter = 0 # Used for triggering timeout
        self.t = 0 # represents the actual time

        self.reset()

        self.reward = 0 # this is to store most recent reward
        self.action_lb, self.action_ub = self.quadruped.get_joint_position_bounds()
        self.action_space = spaces.Box(
            low=self.action_lb,
            high=self.action_ub,
            dtype=np.float32
            )

        observation_lb, observation_ub = self._find_space_limits()
        self.observation_space = spaces.Box(
            low=observation_lb,
            high=observation_ub,
            dtype=np.float32
            )


        # self._stone_ids = []
        # self._fake_stone_ids = [] # this is for the self.fake_client simulation instance
        

    def step(self, action):
        if not ((self.action_lb <= action) & (action <= self.action_ub)).all():
            print("Action passed to env.step(): ", action)
            raise ValueError('Action is out-of-bounds of:\n' + str(self.action_lb) + '\nto\n' + str(self.action_ub)) 
            
        self.quadruped.set_joint_position_targets(action)

        if (np.random.rand() > self.perturbation_rate) and self._apply_perturbations: 
            raise NotImplementedError
            self._apply_perturbation()
        for _ in range(self.n_hold_frames):
            self.client.stepSimulation()
        self.eps_step_counter += 1
        self._update_state()
        done, info = self._is_state_terminal()
        self.reward = self._reward_function()

        # info = {'':''} # this is returned so that env.step() matches Open AI gym API
        if done:
            self.eps_step_counter = 0
            # if self.trot_prior:
                # info['avg_trot_loss'] = self.trot_loss_history.mean()
                # self.trot_loss_history = np.array([])
        return self.state, self.reward, done, info

        
    def reset(self): #TODO add stochasticity
        '''Resets the robot to a neutral standing position, knees slightly bent. The motor control command is to 
        prevent the robot from jumping/falling on first user command. Simulation is stepped to allow robot to fall
        to ground and settle completely.'''

        self.client.resetSimulation()
        self.fake_client.resetSimulation() 
       
        self.client.setTimeStep(1/240.)
        self.client.setGravity(0,0,-9.8)
        self.client.setRealTimeSimulation(self.realTime) # this has no effect in DIRECT mode, only GUI mode
        
        self.plane = self.client.loadURDF(os.path.join(os.path.dirname(__file__), '../urdf/plane.urdf'))
        self.fake_plane = self.client.loadURDF(os.path.join(os.path.dirname(__file__), '../urdf/plane.urdf'))
        self._create_stepping_stones()
        self.quadruped = aliengo.Aliengo(pybullet_client=self.client, 
                                        max_torque=self.max_torque, 
                                        kp=self.kp, 
                                        kd=self.kd)


        
        self.client.resetBasePositionAndOrientation(self.quadruped.quadruped,
                                            posObj=[0,0,self.height + 0.48], 
                                            ornObj=[0,0,0,1.0]) 

        self.quadruped.reset_joint_positions(stochastic=True) # will put all joints at default starting positions
        for i in range(500): # to let the robot settle on the ground.
            self.client.stepSimulation()
        self._update_state()
        
        return self.state
        
    
    # def _remove_stepping_stones(self):
    #     '''Removes the stepping stones in the fake_client and in the client'''

    #     for _id in self._stone_ids:
    #         self.client.removeBody(_id)
    #     self._stone_ids = []

    #     for _id in self._fake_stone_ids:
    #         self.fake_client.removeBody(_id)
    #     self._fake_stone_ids = []


    def _create_stepping_stones(self):
        '''Creates an identical set of stepping stones in client and fake_client.'''

        # Randomly generate stone locations and heights 
        n_stones = int(self.course_length * self.course_width * self.stone_density)
        stone_heights = (np.random.rand(n_stones) - 0.5) * self.stone_height_range + self.height/2.0 
        stone_x = np.random.rand(n_stones) * self.course_length + 1.0
        stone_y = (np.random.rand(n_stones) - 0.5) * self.course_width

        for client in [self.client, self.fake_client]:
            start_block = client.createCollisionShape(p.GEOM_BOX, 
                                                halfExtents=[1, self.course_width/2.0, self.height/2.0])
            stepping_stone = client.createCollisionShape(p.GEOM_BOX, 
                                                halfExtents=[self.stone_length/2.0, self.stone_length/2.0, self.height/2.0])
            start_body = client.createMultiBody(baseCollisionShapeIndex=start_block, 
                                            basePosition=[0,0,self.height/2.0])
            end_body = client.createMultiBody(baseCollisionShapeIndex=start_block, 
                                            basePosition=[self.course_length + 2.0, 0, self.height/2.],)
            
            # if client == self.client:
            #     self._stone_ids = [start_body, end_body]
            # else:
            #     self._fake_stone_ids = [start_body, end_body]
            for i in range(n_stones):
                _id = client.createMultiBody(baseCollisionShapeIndex=stepping_stone, 
                                        basePosition=[stone_x[i], stone_y[i], stone_heights[i]])
                # if client == self.client:                      
                #     self._stone_ids.append(_id)
                # else:
                #     self._fake_stone_ids.append(_id)

        
    def render(self, mode='human', client=None):
        if client is None: # for some reason I can't use self.client as a default value in the function definition line.
            return self.quadruped.render(mode=mode, client=self.client)
        else:
            return self.quadruped.render(mode=mode, client=client)


    def close(self):
        '''I belive this is required for an Open AI gym env.'''
        pass


    def _find_space_limits(self):
        ''' find upper and lower bounds of action and observation spaces''' 

        torque_lb, torque_ub = self.quadruped.get_joint_torque_bounds()
        position_lb, position_ub = self.quadruped.get_joint_position_bounds()
        velocity_lb, velocity_ub = self.quadruped.get_joint_velocity_bounds()
        observation_lb = np.concatenate((torque_lb, 
                                        position_lb,
                                        velocity_lb, 
                                        -0.78 * np.ones(4), # this is for base orientation in quaternions
                                        np.zeros(4), # foot normal forces
                                        -1e5 * np.ones(3), # cartesian acceleration (arbitrary bound)
                                        -1e5 * np.ones(3), # angular velocity (arbitrary bound)
                                        -np.ones(self.grid_len**2) * self.ray_start_height)) # 5 is a safe arbitrary value 

        observation_ub = np.concatenate((torque_ub, 
                                        position_ub, 
                                        velocity_ub, 
                                        0.78 * np.ones(4),
                                        1e4 * np.ones(4), # arbitrary bound
                                        1e5 * np.ones(3),
                                        1e5 * np.ones(3),
                                        np.ones(self.grid_len**2) * self.ray_start_height))


        return observation_lb, observation_ub
            

    def _reward_function(self) -> float:
        ''' Calculates reward based off of current state '''

        base_x_velocity = self.base_twist[0]
        torque_penalty = np.power(self.applied_torques, 2).mean()
        finish_bonus = (self.base_position[0] >= self.course_length + 2) * 200 
        return base_x_velocity - 0.000005 * torque_penalty + finish_bonus



    def _is_state_terminal(self) -> bool:
        ''' Calculates whether to end current episode due to failure based on current state. '''

        info = {}

        fell = self.base_position[2] <= (self.height - self.stone_height_range/2.0)
        reached_goal = (self.base_position[0] >= self.course_length + 2)
        timeout = (self.eps_step_counter >= self.eps_timeout)
        # I don't care about how much the robot yaws for termination, only if its flipped on its back.
        flipping = ((abs(np.array(p.getEulerFromQuaternion(self.base_orientation))) > [0.78*2, 0.78*2.5, 1e10]).any())

        if flipping:
            info['termination_reason'] = 'flipping'
        elif fell:
            info['termination_reason'] = 'fell'
        elif reached_goal:
            info['termination_reason'] = 'reached_goal'
        elif timeout: # {'TimeLimit.truncated': True}
            info['TimeLimit.truncated'] = True

        return any([fell, reached_goal, timeout, flipping]), info


    def _update_state(self):

        self.joint_positions, self.joint_velocities, _, self.applied_torques = self.quadruped.get_joint_states()
        self.base_position, self.base_orientation = self.quadruped.get_base_position_and_orientation()
        self.base_twist = self.quadruped.get_base_twist()
        self.cartesian_base_accel = self.base_twist[:3] - self.previous_base_twist[:3] # TODO divide by timestep or assert timestep == 1/240.

        self.t = self.eps_step_counter * self.n_hold_frames / 240.

        self.foot_normal_forces = self.quadruped._get_foot_contacts()
        
        self.state = np.concatenate((self.applied_torques, 
                                    self.joint_positions,
                                    self.joint_velocities,
                                    self.base_orientation,
                                    self.foot_normal_forces,
                                    self.cartesian_base_accel,
                                    self.base_twist[3:], # base angular velocity
                                    self.quadruped._get_heightmap(self.fake_client, 
                                                                    self.ray_start_height, 
                                                                    self.base_position,
                                                                    self.heightmap_params).flatten())) 
        
        if np.isnan(self.state).any():
            print('nans in state')
            breakpoint()

        # Not used in state, but used in _is_terminal() and _reward()    
        self.previous_base_twist = self.base_twist
    

if __name__ == '__main__':
    '''This test open the simulation in GUI mode for viewing the generated terrain, then saves a rendered image of each
    client for visual verification that the two are identical. There are two resets to ensure that the deletion and 
    addition of terrain elements is working properly. '''


    # import pybullet_envs
    # env = gym.make('gym_aliengo:AliengoSteppingStones-v0', render=False, realTime=False)
    # # env = gym.make('MinitaurBulletEnv-v0')
    # for _ in range(1000):
    #     # env._create_stepping_stones()
    #     # env._remove_stepping_stones()
    #     # p.resetSimulation(env.client)
    #     # p.resetSimulation(env.fake_client)
    #     env.reset()
    # sys.exit()


    env = gym.make('gym_aliengo:AliengoSteppingStones-v0', render=True, realTime=True)
    env.reset()
    imwrite('client_render.png', cvtColor(env.render(client=env.client, mode='rgb_array'), COLOR_RGB2BGR))
    imwrite('fake_client_render.png', cvtColor(env.render(client=env.fake_client, mode='rgb_array'), COLOR_RGB2BGR))

    while True:
        time.sleep(1)


