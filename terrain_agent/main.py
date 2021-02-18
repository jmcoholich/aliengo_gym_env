import torch
import numpy as np
import pybullet as p
import gym
import time
import os 
import matplotlib.pyplot as plt
from observation import get_observation
from agent import TerrainAgent


N_ENVS = 2
NUM_X = 2 # number of footstep placements along x direction, per env
NUM_Y = 2 # number of footstep placements along y direction, per env
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

    # initialize neural network
    agent = TerrainAgent()

    heightmap_params = {'length': 1.25, # TODO allow rectangular
                        'robot_position': 0.5, 
                        'grid_spacing': 1.0/64.0}

    # generate all data in the beginning
    foot_positions, foot, heightmaps, _, _ = get_observation(NUM_X, NUM_Y, envs, heightmap_params=heightmap_params)

    # train
    # TODO normalize the inputs and outputs to the neural network. TODO make/plot the outputs of the nn as relative to the height of the torso.
    # TODO check over the agent to make sure everything is correctly relative
    # TODO use float 32 instead of float 64 for everything.
    # TODO make sure the envs are freed from memory after I'm done generating data
    # TODO add CUDA
    # TODO add noise to the observations
    
    # fwd pass
    foot_positions = torch.from_numpy(foot_positions).type(torch.float32)
    foot = torch.from_numpy(foot).unsqueeze(1).type(torch.float32)
    heightmaps = torch.from_numpy(heightmaps).unsqueeze(1).type(torch.float32) # add channel dimension of 1
    breakpoint()
    output = agent(foot_positions, foot, heightmaps)
    



    # backward pass   

if __name__ == '__main__':
    main()












