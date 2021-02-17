import torch
import numpy as np
import pybullet as p
import gym
import time
import os 
import matplotlib.pyplot as plt
from observation import get_observation


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


    output, foot = get_observation(NUM_X, NUM_Y, envs)

    
            

if __name__ == '__main__':
    main()












