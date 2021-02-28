import torch
import numpy as np
import pybullet as p
import gym
import time
import os 
import matplotlib.pyplot as plt
from observation import get_observation
from agent import TerrainAgent
from loss import loss


# N_ENVS = 1
# NUM_X = 7 # number of footstep placements along x direction, per env
# NUM_Y = 3 # number of footstep placements along y direction, per env
# np.random.seed(1)

def eval_agent(agent, config, vis=False): #TODO finish this shit

    # create several envs of varying difficulty
    envs = [''] * config.n_envs.test 
    for i in range(len(envs)):
        envs[i] = gym.make(config.env, 
                        rows_per_m=np.random.uniform(1.0, 5.0), 
                        terrain_height_range=np.random.uniform(0.0, 0.375), 
                        render=True,
                        fixed=True,
                        fixed_position=[-10,0,1.0])


    # generate all data in the beginning
    foot_positions, foot, heightmaps, x_pos, y_pos, est_robot_base_height, _ = get_observation(num_x, num_y, envs, 
                                                                    heightmap_params=heightmap_params, vis=vis)


    loss = Loss(envs, device)
    loss.to(device)
    del envs

    x_pos = torch.from_numpy(x_pos).type(torch.float32).to(device)
    y_pos = torch.from_numpy(y_pos).type(torch.float32).to(device)
    est_robot_base_height = torch.from_numpy(est_robot_base_height).type(torch.float32).to(device)
    env_idx = torch.from_numpy(env_idx).type(torch.float32).to(device)

    # initialize neural network
    torch.set_default_dtype(torch.float32)
    foot_positions = torch.from_numpy(foot_positions).type(torch.float32).to(device)
    foot = torch.from_numpy(foot).unsqueeze(1).type(torch.float32).to(device)
    heightmaps = torch.from_numpy(heightmaps).unsqueeze(1).type(torch.float32).to(device) # add channel dimension of 1

    means = [foot_positions.mean(axis=0, keepdims=True), foot.mean(), heightmaps.mean()]
    if foot.shape[0] == 1: raise RuntimeError('batch size must be greater than one in order to calculate std')
    stds = [foot_positions.std(axis=0, keepdims=True), foot.std(), heightmaps.std()]
    agent = TerrainAgent(means=means, stds=stds).to(device)
    wandb.watch(agent)


    # fwd pass
    foot_positions = torch.from_numpy(foot_positions).type(torch.float32)
    foot = torch.from_numpy(foot).unsqueeze(1).type(torch.float32)
    heightmaps = torch.from_numpy(heightmaps).unsqueeze(1).type(torch.float32) # add channel dimension of 1
    output = agent(foot_positions, foot, heightmaps)
    output = output.detach().numpy()
    if vis:
        pred_foot_shp = envs[0].client.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=[1., 1., 1., 1.])
        for i in range(len(output)):
            basePosition = [output[i,0] + x_pos[i], output[i,1] + y_pos[i], output[i,2] + est_robot_base_height[i]]
            envs[0].client.createMultiBody(baseVisualShapeIndex=pred_foot_shp, basePosition=basePosition)
        time.sleep(1e5)


    # backward pass   

if __name__ == '__main__':
    eval_agent(None, vis=True)


