import torch
import numpy as np
import pybullet as p
import gym
import time
import os 
import matplotlib.pyplot as plt
from observation import get_observation
from agent import TerrainAgent


N_ENVS = 1
NUM_X = 4 # number of footstep placements along x direction, per env
NUM_Y = 3 # number of footstep placements along y direction, per env
# np.random.seed(1)

def eval_agent(agent, vis=False):

    # create several envs of varying difficulty
    envs = [''] * N_ENVS
    for i in range(len(envs)):
        envs[i] = gym.make('gym_aliengo:AliengoSteps-v0', 
                        rows_per_m=np.random.uniform(1.0, 5.0), 
                        terrain_height_range=np.random.uniform(0, 0.375), render=True,
                        fixed=True,
                        fixed_position=[-10,0,1.0],
                        terrain_width=3.0,
                        terrain_length=5.0)

    # initialize neural network
    agent = TerrainAgent()

    heightmap_params = {'length': 1.25, # TODO allow rectangular
                        'robot_position': 0.5, 
                        'grid_spacing': 1.0/64.0}

    # generate all data in the beginning
    foot_positions, foot, heightmaps, x_pos, y_pos = get_observation(NUM_X, NUM_Y, envs, 
                                                                    heightmap_params=heightmap_params, vis=vis)

    
    # fwd pass
    foot_positions = torch.from_numpy(foot_positions).type(torch.float32)
    foot = torch.from_numpy(foot).unsqueeze(1).type(torch.float32)
    heightmaps = torch.from_numpy(heightmaps).unsqueeze(1).type(torch.float32) # add channel dimension of 1
    output = agent(foot_positions, foot, heightmaps)
    output = output.detach().numpy()
    if vis:
        pred_foot_shp = envs[0].client.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=[1., 1., 1., 1.])
        for i in range(len(output)):
            basePosition = [output[i,0] + x_pos[i], output[i,1] + y_pos[i], output[i,2]]
            envs[0].client.createMultiBody(baseVisualShapeIndex=pred_foot_shp, basePosition=basePosition)
        time.sleep(1e5)


    # backward pass   

if __name__ == '__main__':
    eval_agent(None, vis=True)


