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

'''
TODO
- use createCollisionShapeArray
'''

class AliengoSteppingStones(_aliengo_parent.AliengoEnvParent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # stepping stone parameters
        self.height = 1.0 # height of the heightfield
        self.course_length = 5.0 # total distance from edge of start block to edge of end block 
        self.course_width = 2.0 # widght of path of stepping stones 
        self.stone_length = 0.25 # side length of square stepping stones
        self.stone_density = 6.0 # stones per square meter 
        self.stone_height_range = 0.25 # heights of stones will be within [self.height - this/2, self.height + this/2 ]
        self.ray_start_height = self.height + self.stone_height_range

        self.reset()


    def reset(self):
        super()._hard_reset()
        self._create_stepping_stones()
        self.quadruped = aliengo.Aliengo(pybullet_client=self.client, 
                                        max_torque=self.max_torque, 
                                        kp=self.kp, 
                                        kd=self.kd)
        return super().reset(base_height=self.height + 0.48)


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
            
            for i in range(n_stones):
                _id = client.createMultiBody(baseCollisionShapeIndex=stepping_stone, 
                                        basePosition=[stone_x[i], stone_y[i], stone_heights[i]])


    def _is_state_terminal(self):
        ''' Calculates whether to end current episode due to failure based on current state. '''

        quadruped_done, termination_dict = self.quadruped.is_state_terminal(flipping_bounds=[np.pi/2.0]*3,
                                                            height_lb=self.height - self.stone_height_range/2.0,
                                                            height_ub=self.height - self.stone_height_range/2.0 + 1.0) 
        timeout = (self.eps_step_counter >= self.eps_timeout) or \
                    (self.base_position[0] >= self.course_length + 2.0)
        # the height termination condition should take care of y_out_of_bounds
        # y_out_of_bounds = not (-self.stairs_width/2. < self.base_position[1] < self.stairs_width/2.)
        if timeout:
            termination_dict['TimeLimit.truncated'] = True
        done = quadruped_done or timeout
        return done, termination_dict


if __name__ == '__main__':
    '''This test open the simulation in GUI mode for viewing the generated terrain, then saves a rendered image of each
    client for visual verification that the two are identical. There are two resets to ensure that the deletion and 
    addition of terrain elements is working properly. '''

    env = gym.make('gym_aliengo:AliengoSteppingStones-v0', render=True, realTime=True)
    env.reset()
    imwrite('client_render.png', cvtColor(env.render(client=env.client, mode='rgb_array'), COLOR_RGB2BGR))
    imwrite('fake_client_render.png', cvtColor(env.render(client=env.fake_client, mode='rgb_array'), COLOR_RGB2BGR))

    while True:
        env.reset()
        time.sleep(5)


