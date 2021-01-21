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
from gym_aliengo.envs import _aliengo_parent 
from pybullet_utils import bullet_client as bc
from math import ceil

'''
Env for steps (random "grid" of elevated rectangles), meant to replicate the Steps env used in this paper: 
https://robotics.sciencemag.org/content/robotics/5/47/eabc5986.full.pdf
'''

class AliengoSteps(_aliengo_parent.AliengoEnvParent):
    def __init__(self,
                row_width=0.5, # range from [0.1, 1.0] (hard to easy) default=0.5
                terrain_height_range=0.25, # range from [0.0, 0.375] (easy to hard) default=0.25
                **kwargs):
        
        super().__init__(**kwargs)
        
        # Terrain parameters, all units in meters
        assert row_width > 0.01
        self.row_width = row_width
        self.terrain_height_range = terrain_height_range # +/- half of this value to the height mean 
        self.terrain_length = 5 #TODO change this back to 50, and make this parameter scale with reward because terrain
        # generation takes a lot of time 
        self.terrain_width = 3 
        self.terrain_height = terrain_height_range/2. + 0.05 # this is just the mean height of the blocks

        self.block_length_range = self.row_width/2. # the mean is set to the same as row_width. 
        self.ramp_distance = self.terrain_height * 4
        self.ray_start_height = self.terrain_height + self.terrain_height_range/2. + 1.

        self.reset()


    def reset(self):
        '''Resets the robot to a neutral standing position, knees slightly bent. The motor control command is to 
        prevent the robot from jumping/falling on first user command. Simulation is stepped to allow robot to fall
        to ground and settle completely.'''

        super()._hard_reset()
        self._create_steps()
        self.quadruped = aliengo.Aliengo(pybullet_client=self.client, 
                                        max_torque=self.max_torque, 
                                        kp=self.kp, 
                                        kd=self.kd)
        return super().reset()


    def _create_steps(self):
        '''Creates an identical steps terrain in client and fake client'''
        
        # pick a list of discrete values for heights to only generate a finite number of collision shapes
        n_shapes = 10
        height_values = np.linspace(-self.terrain_height_range/2., self.terrain_height_range/2., n_shapes)
        length_values = np.linspace(self.row_width - self.block_length_range/2.,
                                    self.row_width + self.block_length_range/2.,
                                    n_shapes)      
        shapeId = np.zeros(n_shapes, dtype=np.int)
        fake_shapeId = np.zeros(n_shapes, dtype=np.int)
        for i in range(len(length_values)):
            halfExtents=[length_values[i]/2., self.row_width/2., self.terrain_height/2.] 
            shapeId[i] = self.client.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents)
            fake_shapeId[i] = self.fake_client.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents)

        for row in range(int(ceil(self.terrain_width/self.row_width))):
            total_len = 0
            while total_len < self.terrain_length:
                i = np.random.randint(0, n_shapes)
                j = np.random.randint(0, n_shapes)
                if total_len < self.ramp_distance:
                    offset = self.terrain_height * (1 - float(total_len)/self.ramp_distance)
                else:
                    offset = 0
                pos = [total_len + length_values[i]/2. + 0.5, # X
                        row * self.row_width - self.terrain_width/2. + self.row_width/2., # Y
                        height_values[j] - offset + self.terrain_height/2.] # Z
                self.client.createMultiBody(baseCollisionShapeIndex=shapeId[i], basePosition=pos)
                self.fake_client.createMultiBody(baseCollisionShapeIndex=fake_shapeId[i], basePosition=pos)
                total_len += length_values[i]


    def _is_state_terminal(self) -> bool:
        ''' Calculates whether to end current episode due to failure based on current state. '''

        quadruped_done, termination_dict = self.quadruped.is_state_terminal(flipping_bounds=[np.pi/2.0]*3, 
                                height_ub=base_z_position > 0.8 + self.terrain_height + self.terrain_height_range/2.) 
        timeout = (self.eps_step_counter >= self.eps_timeout) or \
                    (self.base_position[0] >= self.terrain_length - 1.0)
        y_out_of_bounds = not (-self.terrain_width/2. < self.base_position[1] < self.terrain_width/2.)

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

    env = gym.make('gym_aliengo:AliengoSteps-v0', render=True, realTime=True)
    imwrite('client_render.png', cvtColor(env.render(client=env.client, mode='rgb_array'), COLOR_RGB2BGR))
    imwrite('fake_client_render.png', cvtColor(env.render(client=env.fake_client, mode='rgb_array'), COLOR_RGB2BGR))

    
    while True:
        env.reset()
        time.sleep(5.0)


