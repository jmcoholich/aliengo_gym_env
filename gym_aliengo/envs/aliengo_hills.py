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
from noise import pnoise2
from random import randint

'''
Env for rolling hills, meant to replicate the Hills env used in this paper: 
https://robotics.sciencemag.org/content/robotics/5/47/eabc5986.full.pdf
'''
class AliengoHills(gym.Env):
    def __init__(self, render=False, realTime=False,
                scale=1.0, # good values range from 5.0 (easy) to 0.5 (hard)
                amplitude=0.75): # try [0.1, 1.0]
        
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

        # Hills parameters, all units in meters
        self.hills_height = amplitude
        self.mesh_res = 3 # int, points/meter
        self.hills_length = 50
        self.hills_width = 3
        self.ramp_distance = 1.0
        self.ray_start_height = self.hills_height + 1.0

        # heightmap param dict
        self.heightmap_params = {'length': 1.25, # assumes square 
                            'robot_position': 0.5, # distance of robot base origin from back edge of height map
                            'grid_spacing': 0.125}
        assert self.heightmap_params['length'] % self.heightmap_params['grid_spacing'] == 0
        self.grid_len = int(self.heightmap_params['length']/self.heightmap_params['grid_spacing']) + 1

        # this is a random id appened to terrain file name, so that each env instance doesn't overwrite another one.
        # use randint for filenames, since the np random seed is set, all env instances will get the same random number,
        #  causing them to all write to the same file.
        self.env_terrain_id = randint(0, 1e18) 
        self.path = os.path.join(os.path.dirname(__file__),
                                    '../meshes/generated_hills_' + str(self.env_terrain_id) + '.obj')
        # self.vhacd_path = os.path.join(os.path.dirname(__file__),
        #                             '../meshes/VHACD_generated_hills_' + str(self.env_terrain_id) + '.obj')
        # self.log_path = os.path.join(os.path.dirname(__file__),
        #                             '../meshes/log_VHACD_generated_hills_' + str(self.env_terrain_id) + '.txt')

        

        # Perlin Noise parameters
        self.scale = self.mesh_res * scale
        self.octaves = 1
        self.persistence = 0.0 # roughness basically (assuming octaves > 1). I'm not using this.
        self.lacunarity = 2.0
        self.base = 0 # perlin noise base, to be randomized

        if self.scale == 1.0: # this causes terrain heights of all zero to be returned, for some reason
            self.scale = 1.01

        if self._is_render:
            self.client = bc.BulletClient(connection_mode=p.GUI)
        else:
            self.client = bc.BulletClient(connection_mode=p.DIRECT)
        self.fake_client = bc.BulletClient(connection_mode=p.DIRECT) # this is only used for getting the heightmap 

        self.client.setPhysicsEngineParameter(enableFileCaching=0) # load the newly generated terrain every reset()
        self.fake_client.setPhysicsEngineParameter(enableFileCaching=0)

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
        self._create_hills()
        self.quadruped = aliengo.Aliengo(pybullet_client=self.client, 
                                        max_torque=self.max_torque, 
                                        kp=self.kp, 
                                        kd=self.kd)
        # for link in self.quadruped.foot_links:
        #     self.client.changeDynamics(self.quadruped.quadruped, link, contactStiffness=1e7, contactDamping=1e7)
        #     print('stiffness:',self.client.getDynamicsInfo(self.quadruped.quadruped, link)[9], 
        #             'damping:', self.client.getDynamicsInfo(self.quadruped.quadruped, link)[8])

        self.client.resetBasePositionAndOrientation(self.quadruped.quadruped,
                                            posObj=[0,0,0.48], 
                                            ornObj=[0,0,0,1.0]) 

        self.quadruped.reset_joint_positions(stochastic=True) # will put all joints at default starting positions
        for i in range(500): # to let the robot settle on the ground.
            self.client.stepSimulation()
        self._update_state()
        
        return self.state
    

    def _create_hills(self):
        '''Creates an identical hills mesh using Perlin noise. Added to client and fake client'''
        
        mesh_length = self.hills_length * self.mesh_res
        mesh_width = self.hills_width * self.mesh_res

        vertices = np.zeros((mesh_length + 1, mesh_width + 1))
        self.base = np.random.randint(300)
        for i in range(mesh_length + 1):
            for j in range(mesh_width + 1):
                vertices[i, j] = pnoise2(float(i)/(self.scale),
                                            float(j)/(self.scale),
                                            octaves=self.octaves,
                                            persistence=self.persistence,
                                            lacunarity=self.lacunarity,
                                            repeatx=mesh_length + 1,
                                            repeaty=mesh_width + 1,
                                            base=self.base) # base is the seed
        # Uncomment below to visualize image of terrain map                                            
        # from PIL import Image
        # Image.fromarray(((np.interp(vertices, (vertices.min(), vertices.max()), (0, 255.0))>128)*255).astype('uint8'), 'L').show()
        vertices = np.interp(vertices, (vertices.min(), vertices.max()), (0, 1.0))


        # ramp down n meters, so the robot can walk up onto the hills terrain
        for i in range(int(self.ramp_distance * self.mesh_res)):
            vertices[i, :] *= i/(self.ramp_distance * self.mesh_res)
        vertices = vertices * self.hills_height # terrain height

        with open(self.path,'w') as f:
            f.write('o Generated_Hills_Terrain_' + str(self.env_terrain_id) + '\n')
            # write vertices
            for i in range(mesh_length + 1):
                for j in range(mesh_width + 1):
                    f.write('v  {}   {}   {}\n'.format(i, j, vertices[i, j]))

            # write faces 
            for i in range(mesh_length):
                for j in range(mesh_width):
                    # bottom left triangle
                    f.write('f  {}   {}   {}\n'.format((mesh_width + 1)*i + j+1, 
                                                        (mesh_width + 1)*i + j+2, 
                                                        (mesh_width + 1)*(i+1) + j+1)) 
                    # top right triangle
                    f.write('f  {}   {}   {}\n'.format((mesh_width + 1)*(i+1) + j+2, 
                                                        (mesh_width + 1)*(i+1) + j+1, 
                                                        (mesh_width + 1)*i + j+2)) 
                    # repeat, making faces double-sided
                    f.write('f  {}   {}   {}\n'.format((mesh_width + 1)*i + j+2, 
                                                        (mesh_width + 1)*i + j+1, 
                                                        (mesh_width + 1)*(i+1) + j+1)) 
                    f.write('f  {}   {}   {}\n'.format((mesh_width + 1)*(i+1) + j+1, 
                                                        (mesh_width + 1)*(i+1) + j+2, 
                                                        (mesh_width + 1)*i + j+2)) 
        # self.client.vhacd(self.path, self.vhacd_path, self.log_path)                          
        terrain = self.client.createCollisionShape(p.GEOM_MESH, 
                                                    meshScale=[1.0/self.mesh_res, 1.0/self.mesh_res, 1.0], 
                                                    fileName=self.path,
                                                    flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        fake_terrain = self.fake_client.createCollisionShape(p.GEOM_MESH, 
                                                    meshScale=[1.0/self.mesh_res, 1.0/self.mesh_res, 1.0], 
                                                    fileName=self.path,
                                                    flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        
        ori = self.client.getQuaternionFromEuler([0, 0, 0])
        pos = [0.5 , -self.hills_width/2, 0]
        self.client.createMultiBody(baseCollisionShapeIndex=terrain, baseOrientation=ori, basePosition=pos)
        self.fake_client.createMultiBody(baseCollisionShapeIndex=fake_terrain, baseOrientation=ori, basePosition=pos)

        
        
        # print('stiffness:',self.client.getDynamicsInfo(self.plane, -1)[9], 
        #         'damping:', self.client.getDynamicsInfo(self.plane, -1)[8])
        # print('stiffness:',self.client.getDynamicsInfo(terrain, -1)[9], 
        #         'damping:', self.client.getDynamicsInfo(terrain, -1)[8])

        

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
        return base_x_velocity - 0.000005 * torque_penalty



    def _is_state_terminal(self) -> bool:
        ''' Calculates whether to end current episode due to failure based on current state. '''

        info = {}

        base_z_position = self.base_position[2]
        height_out_of_bounds = ((base_z_position < 0.1) or (base_z_position > 0.9 + self.hills_height))
        timeout = (self.eps_step_counter >= self.eps_timeout) or \
                    (self.base_position[0] >= self.hills_length - 0.5) # don't want it to fall off the end.
        # I don't care about how much the robot yaws for termination, only if its flipped on its back.
        flipping = ((abs(np.array(p.getEulerFromQuaternion(self.base_orientation))) > [0.78*2, 0.78*2.5, 1e10]).any())
        y_out_of_bounds = not (-self.hills_width/2. < self.base_position[1] < self.hills_width/2.)

        if flipping:
            info['termination_reason'] = 'flipping'
        elif height_out_of_bounds:
            info['termination_reason'] = 'height_out_of_bounds'
        elif y_out_of_bounds:
            info['termination_reason'] = 'y_out_of_bounds'
        elif timeout: # {'TimeLimit.truncated': True}
            info['TimeLimit.truncated'] = True

        return any([flipping, height_out_of_bounds, timeout, y_out_of_bounds]), info


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

    env = gym.make('gym_aliengo:AliengoHills-v0', render=True, realTime=True)
    imwrite('client_render.png', cvtColor(env.render(client=env.client, mode='rgb_array'), COLOR_RGB2BGR))
    imwrite('fake_client_render.png', cvtColor(env.render(client=env.fake_client, mode='rgb_array'), COLOR_RGB2BGR))

    while True:
        env.reset()
        time.sleep(1.0)


