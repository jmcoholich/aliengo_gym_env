import torch
import numpy as np
import pybullet as p
import gym
import time
import os 
import matplotlib.pyplot as plt
from gen_footsteps import rand_footsteps


N_ENVS = 2
NUM_X = 4 # number of footstep placements along x direction, per env
NUM_Y = 3 # number of footstep placements along y direction, per env
# np.random.seed(1)


# TODO: 

def main():

    # create several envs of varying difficulty
    envs = [''] * N_ENVS
    for i in range(len(envs)):
        envs[i] = gym.make('gym_aliengo:AliengoSteps-v0', 
                        rows_per_m=np.random.uniform(1.0, 5.0), 
                        terrain_height_range=np.random.uniform(0, 0.375), render=False,
                        fixed=True,
                        fixed_position=[-10,0,1.0])
    assert envs[i].terrain_length == 20  
    assert envs[i].terrain_width == 10 


    # generate footstep placements in each environment
    x_positions = np.linspace(0.0, envs[0].terrain_length - 1.0, NUM_X)
    y_positions = np.linspace(-envs[0].terrain_width/2.0 + 0.5, envs[0].terrain_width/2.0 - 0.5, NUM_Y)
    output, foot = rand_foosteps(x_positions, y_positions, envs)

    # get the heightmap around each x and y position. Heighmaps will go through CNN encoder, so store as 2D arrays
    heightmap_params = {'length': 1.25, # assumes square #TODO allow rectangular
                            'robot_position': 0.5, # distance of robot base origin from back edge of height map
                            'grid_spacing': 0.125}
    pts_per_env = NUM_X * NUM_Y
    assert pts_per_env * N_ENVS == len(output)
    heightmaps = np.zeros((pts_per_env * N_ENVS, self.heightmap_params['length']**2))
    for i in range(N_ENVS):
        for j in range(pts_per_env * i, pts_per_env * (i + 1)):
            heightmaps[j] = envs[i].quadruped._get_heightmap(envs[i].fake_client, 
                                                        ray_start_height=100, #TODO 
                                                        base_position=[], #TODO
                                                        heightmap_params=heightmap_params)
            

if __name__ == '__main__':
    main()












