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
from gym_aliengo.envs import aliengo_env
from pybullet_utils import bullet_client as bc
from noise import pnoise2
from random import randint

'''
Env for rolling hills, meant to replicate the Hills env used in this paper: 
https://robotics.sciencemag.org/content/robotics/5/47/eabc5986.full.pdf
'''

class AliengoHills(aliengo_env.AliengoEnv):

    def __init__(self, 
                scale=1.0, # good values range from 5.0 (easy) to 0.5 (hard)
                amplitude=0.75, # try [0.1, 1.0]
                **kwargs): 

        super().__init__(**kwargs)

        # Hills parameters, all units in meters
        self.hills_height = amplitude
        self.mesh_res = 15 # int, points/meter
        self.hills_length = 50
        self.hills_width = 5
        self.ramp_distance = 1.0
        self.ray_start_height = self.hills_height + 1.0

        # Perlin Noise parameters
        self.scale = self.mesh_res * scale
        self.octaves = 1
        self.persistence = 0.0 # roughness basically (assuming octaves > 1). I'm not using this.
        self.lacunarity = 2.0
        self.base = 0 # perlin noise base, to be randomized
        self.terrain = None # to be set later
        self.fake_terrain = None # to be set later

        if self.scale == 1.0: # this causes terrain heights of all zero to be returned, for some reason
            self.scale = 1.01

        #TODO consider adding this to another parent from HERE
        self.fake_client = bc.BulletClient(connection_mode=p.DIRECT) 
        if self.fake_client == -1:
            raise RuntimeError('Pybullet could not connect to physics client')
        # heightmap param dict 
        self.heightmap_params = {'length': 1.25, # assumes square 
                            'robot_position': 0.5, # distance of robot base origin from back edge of height map
                            'grid_spacing': 0.125}
        assert self.heightmap_params['length'] % self.heightmap_params['grid_spacing'] == 0
        self.grid_len = int(self.heightmap_params['length']/self.heightmap_params['grid_spacing']) + 1


        # TODO to here maybe
         
        self.reset(hard_reset=True) # hard reset clears the simulation and creates heightfield from scratch. Its faster

        self.client.setPhysicsEngineParameter(enableFileCaching=0) # load the newly generated terrain every reset()
        self.fake_client.setPhysicsEngineParameter(enableFileCaching=0)

        # to not do a hard reset, but hard reset is necessary in the constructor 

    # def render(self, mode='human', client=None):
    #     if client is None: # for some reason I can't use self.client as a default value in the function definition line.
    #         return self.quadruped.render(mode=mode, client=self.client)
    #     else:
    #         return self.quadruped.render(mode=mode, client=client)

    def reset(self, hard_reset=False):
        '''Resets the robot to a neutral standing position, knees slightly bent. The motor control command is to 
        prevent the robot from jumping/falling on first user command. Simulation is stepped to allow robot to fall
        to ground and settle completely.'''

        if hard_reset:
            # TODO I think I can just put this code block into a super()._hard_reset() function 
            self.client.resetSimulation()
            self.fake_client.resetSimulation() 
        
            self.client.setTimeStep(1/240.)
            self.client.setGravity(0,0,-9.8)
            self.client.setRealTimeSimulation(self.realTime) # this has no effect in DIRECT mode, only GUI mode
            
            self.plane = self.client.loadURDF(os.path.join(os.path.dirname(__file__), '../urdf/plane.urdf'))
            self.fake_plane = self.fake_client.loadURDF(os.path.join(os.path.dirname(__file__), '../urdf/plane.urdf'))
            # UNTIL here #TODO
            self._create_hills(update=False)
            self.quadruped = aliengo.Aliengo(pybullet_client=self.client, 
                                            max_torque=self.max_torque, 
                                            kp=self.kp, 
                                            kd=self.kd)

        else: 
            self._create_hills(update=True)

        return super().reset() # resets quadruped position (and a few other vars), updates state, returns observation
    

    def _create_hills(self, update):
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
        # vertices = vertices * self.hills_height # terrain height
        meshScale = [1.0/self.mesh_res, 1.0/self.mesh_res, self.hills_height]
        heightfieldTextureScaling = self.mesh_res/2.

        if not update:
            self.terrain = self.client.createCollisionShape(p.GEOM_HEIGHTFIELD, 
                                                        meshScale=meshScale, 
                                                        heightfieldTextureScaling=heightfieldTextureScaling,
                                                        heightfieldData=vertices.flatten(),
                                                        numHeightfieldRows=mesh_width + 1,
                                                        numHeightfieldColumns=mesh_length + 1)
            self.fake_terrain = self.fake_client.createCollisionShape(p.GEOM_HEIGHTFIELD, 
                                                        meshScale=meshScale, 
                                                        heightfieldTextureScaling=heightfieldTextureScaling,
                                                        heightfieldData=vertices.flatten(),
                                                        numHeightfieldRows=mesh_width + 1,
                                                        numHeightfieldColumns=mesh_length + 1)
        
            
            ori = self.client.getQuaternionFromEuler([0, 0, -np.pi/2.])
            pos = [self.hills_length/2. +0.5 , 0, self.hills_height/2.]
            self.client.createMultiBody(baseCollisionShapeIndex=self.terrain, baseOrientation=ori, basePosition=pos)
            self.fake_client.createMultiBody(baseCollisionShapeIndex=self.fake_terrain, baseOrientation=ori, basePosition=pos)
        
        else: # just update existing mesh
            self.client.createCollisionShape(p.GEOM_HEIGHTFIELD, 
                                                        meshScale=meshScale, 
                                                        heightfieldTextureScaling=heightfieldTextureScaling,
                                                        heightfieldData=vertices.flatten(),
                                                        numHeightfieldRows=mesh_width + 1,
                                                        numHeightfieldColumns=mesh_length + 1,
                                                        replaceHeightfieldIndex=self.terrain)
            self.fake_client.createCollisionShape(p.GEOM_HEIGHTFIELD, 
                                                        meshScale=meshScale, 
                                                        heightfieldTextureScaling=heightfieldTextureScaling,
                                                        heightfieldData=vertices.flatten(),
                                                        numHeightfieldRows=mesh_width + 1,
                                                        numHeightfieldColumns=mesh_length + 1,
                                                        replaceHeightfieldIndex=self.fake_terrain)


    def _is_state_terminal(self):
        ''' Adds condition for running out of terrain.'''

        quadruped_done, termination_dict = self.quadruped.is_state_terminal(flipping_bounds=[np.pi/2.0]*3)
        timeout = (self.eps_step_counter >= self.eps_timeout) or \
                    (self.base_position[0] >= self.hills_length - 0.5) # don't want it to fall off the end.
        y_out_of_bounds = not (-self.hills_width/2. < self.base_position[1] < self.hills_width/2.)
        if timeout:
            termination_dict['TimeLimit.truncated'] = True
        elif y_out_of_bounds:
            # this overwrites previous termination reason in the rare case that more than one occurs at once.
            termination_dict['termination_reason'] = 'y_out_of_bounds'
        done = quadruped_done or timeout or y_out_of_bounds
        return done, termination_dict


if __name__ == '__main__':
    '''This test open the simulation in GUI mode for viewing the generated terrain, then saves a rendered image of each
    client for visual verification that the two are identical. Then the script just keeps generating random terrains 
    for viewing. '''

    env = gym.make('gym_aliengo:AliengoHills-v0', render=True, realTime=True)
    imwrite('client_render.png', cvtColor(env.render(client=env.client, mode='rgb_array'), COLOR_RGB2BGR))
    imwrite('fake_client_render.png', cvtColor(env.render(client=env.fake_client, mode='rgb_array'), COLOR_RGB2BGR))

    while True:
        env.reset(hard_reset=False)
        time.sleep(1.0)

