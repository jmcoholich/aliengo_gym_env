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
from math import ceil

'''
Env for stairs meant to replicate the Steps env used in this paper: 
https://robotics.sciencemag.org/content/robotics/5/47/eabc5986.full.pdf
'''
class AliengoStairs(gym.Env):
    def __init__(self, render=False, realTime=False,
                step_height=0.25, # [0.0, 0.5?] unknown how high we can really go Default = 0.25
                step_length=2.0): # [0.25, 3.0] Default = 2.0

        # Environment Options
        self._apply_perturbations = False
        self.perturbation_rate = 0.00 # probability that a random perturbation is applied to the torso
        self.max_torque = 40.0 
        self.kp = 0.1
        self.kd = 1.0
        self.n_hold_frames = 4 
        self._is_render = render
        self.eps_timeout = 240.0/self.n_hold_frames * 20 # number of steps to timeout after
        self.realTime = realTime

        # Stairs parameters, all units in meters
        self.stairs_length = 50
        self.stairs_width = 10
        self.step_height = step_height # this is a mean
        self.step_length = step_length # this is a mean
        self.step_height_range = self.step_height/2.
        self.step_length_range = self.step_length/2.
        self.ray_start_height = self.stairs_length / self.step_length * self.step_height * 2

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

        # self.client.setPhysicsEngineParameter(enableFileCaching=0) # load the newly generated terrain every reset()
        # self.fake_client.setPhysicsEngineParameter(enableFileCaching=0)

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

        if done:
            self.eps_step_counter = 0

        return self.state, self.reward, done, info

        
    def reset(self):
        '''Resets the robot to a neutral standing position, knees slightly bent. The motor control command is to 
        prevent the robot from jumping/falling on first user command. Simulation is stepped to allow robot to fall
        to ground and settle completely.'''

        self.client.resetSimulation()
        self.fake_client.resetSimulation() 
       
        self.client.setTimeStep(1/240.)
        self.client.setGravity(0,0,-9.8)
        self.client.setRealTimeSimulation(self.realTime) # this has no effect in DIRECT mode, only GUI mode
        
        self.plane = self.client.loadURDF(os.path.join(os.path.dirname(__file__), '../urdf/plane.urdf'))
        self.fake_plane = self.fake_client.loadURDF(os.path.join(os.path.dirname(__file__), '../urdf/plane.urdf'))
        self._create_stairs()
        self.quadruped = aliengo.Aliengo(pybullet_client=self.client, 
                                        max_torque=self.max_torque, 
                                        kp=self.kp, 
                                        kd=self.kd)
        
        self.client.resetBasePositionAndOrientation(self.quadruped.quadruped,
                                            posObj=[0,0,0.48], 
                                            ornObj=[0,0,0,1.0]) 

        self.quadruped.reset_joint_positions(stochastic=True) # will put all joints at default starting positions
        for i in range(500): # to let the robot settle on the ground.
            self.client.stepSimulation()
        self._update_state()
        
        return self.state


    def _create_stairs(self):
        '''Creates an identical steps terrain in client and fake client'''

        # set thickness to largest height so that there are never any gaps in the stairs
        step_thickness = self.step_height + self.step_height_range/2. + 0.01 
        
        # I think I can use the same collision shape for everything
        halfExtents = [(self.step_length + self.step_length_range/2. + 0.01) / 2., 
                        self.stairs_width/2., 
                        step_thickness/2.]
        _id = self.client.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents)
        fake_id = self.fake_client.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents)

        total_len = 0
        total_height = 0
        while total_len < self.stairs_length:
            height = (np.random.rand() - 0.5) * self.step_height_range + self.step_height
            length = (np.random.rand() - 0.5) * self.step_length_range + self.step_length
            pos = [total_len + length/2. + 1.0, 0.0, total_height + height/2.]
            self.client.createMultiBody(baseCollisionShapeIndex=_id, basePosition=pos)
            self.fake_client.createMultiBody(baseCollisionShapeIndex=fake_id, basePosition=pos)
            total_len += length
            total_height += height
               

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
                                        -np.ones(self.grid_len**2) * (self.ray_start_height)))

        observation_ub = np.concatenate((torque_ub, 
                                        position_ub, 
                                        velocity_ub, 
                                        0.78 * np.ones(4),
                                        1e4 * np.ones(4), # arbitrary bound
                                        1e5 * np.ones(3),
                                        1e5 * np.ones(3),
                                        np.ones(self.grid_len**2) * (self.ray_start_height)))


        return observation_lb, observation_ub
            

    def _reward_function(self) -> float:
        ''' Calculates reward based off of current state '''

        base_x_velocity = self.base_twist[0]
        torque_penalty = np.power(self.applied_torques, 2).mean()
        return base_x_velocity - 0.000005 * torque_penalty


    def _is_state_terminal(self) -> bool:
        ''' Calculates whether to end current episode due to failure based on current state. '''

        info = {}

        timeout = (self.eps_step_counter >= self.eps_timeout) or \
                    (self.base_position[0] >= self.stairs_length - 2.0)
        # I don't care about how much the robot yaws for termination, only if its flipped on its back.
        flipping = ((abs(np.array(p.getEulerFromQuaternion(self.base_orientation))) > [0.78*2, 0.78*2.5, 1e10]).any())
        y_out_of_bounds = not (-self.stairs_width/2. < self.base_position[1] < self.stairs_width/2.)

        if flipping:
            info['termination_reason'] = 'flipping'
        elif y_out_of_bounds:
            info['termination_reason'] = 'y_out_of_bounds'
        elif timeout: # {'TimeLimit.truncated': True}
            info['TimeLimit.truncated'] = True

        return any([flipping, timeout, y_out_of_bounds]), info


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
    client for visual verification that the two are identical. Then the script just keeps generating random terrains 
    for viewing. '''

    env = gym.make('gym_aliengo:AliengoStairs-v0', render=True, realTime=True)
    imwrite('client_render.png', cvtColor(env.render(client=env.client, mode='rgb_array'), COLOR_RGB2BGR))
    imwrite('fake_client_render.png', cvtColor(env.render(client=env.fake_client, mode='rgb_array'), COLOR_RGB2BGR))

    
    while True:
        env.reset()
        time.sleep(5.0)


