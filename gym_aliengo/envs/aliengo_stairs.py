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
Env for stairs meant to replicate the Steps env used in this paper: 
https://robotics.sciencemag.org/content/robotics/5/47/eabc5986.full.pdf
'''


class AliengoStairs(_aliengo_parent.AliengoEnvParent):
    def __init__(self, 
                step_height=0.25, # [0.0, 0.5?] unknown how high we can really go Default = 0.25
                step_length=2.0, # [0.25, 3.0] Default = 2.0
                **kwargs):
        
        super().__init__(**kwargs)

        # Stairs parameters, all units in meters
        self.stairs_length = 50
        self.stairs_width = 10
        self.step_height = step_height # this is a mean
        self.step_length = step_length # this is a mean
        self.step_height_range = self.step_height/2.
        self.step_length_range = self.step_length/2.
        self.ray_start_height = self.stairs_length / self.step_length * self.step_height * 2

        self.reset()


    def reset(self):       
        super()._hard_reset()
        self._create_stairs()
        self.quadruped = aliengo.Aliengo(pybullet_client=self.client, 
                                        max_torque=self.max_torque, 
                                        kp=self.kp, 
                                        kd=self.kd,
                                        vis=self.vis)
        return super().reset()


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


    def _is_state_terminal(self):
        ''' Calculates whether to end current episode due to failure based on current state. '''

        quadruped_done, termination_dict = self.quadruped.is_state_terminal(flipping_bounds=[np.pi/2.0]*3, 
                                                                            height_ub=np.inf) # stairs can go very high 
        timeout = (self.eps_step_counter >= self.eps_timeout) or \
                    (self.quadruped.base_position[0] >= self.stairs_length - 2.0)
        y_out_of_bounds = not (-self.stairs_width/2. < self.quadruped.base_position[1] < self.stairs_width/2.)
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

    env = gym.make('gym_aliengo:AliengoStairs-v0', render=True, realTime=True)
    imwrite('client_render.png', cvtColor(env.render(client=env.client, mode='rgb_array'), COLOR_RGB2BGR))
    imwrite('fake_client_render.png', cvtColor(env.render(client=env.fake_client, mode='rgb_array'), COLOR_RGB2BGR))

    
    while True:
        env.reset()
        time.sleep(5.0)


