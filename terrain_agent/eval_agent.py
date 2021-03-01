import torch
import numpy as np
import pybullet as p
import gym
import time
import os 
import matplotlib.pyplot as plt
from observation import get_observation
from agent import TerrainAgent
from loss import Loss


# N_ENVS = 1
# NUM_X = 7 # number of footstep placements along x direction, per env
# NUM_Y = 3 # number of footstep placements along y direction, per env
# np.random.seed(1)

def eval_agent(agent, config, vis=False): 

    # create several envs of varying difficulty
    envs = [''] * config['n_envs_test']
    for i in range(len(envs)):
        envs[i] = gym.make(config['env'], 
                        rows_per_m=np.random.uniform(1.0, 5.0), 
                        terrain_height_range=np.random.uniform(0.0, 0.375), 
                        render=False,
                        fixed=True,
                        fixed_position=[-10,0,1.0])


    # generate all data in the beginning
    foot_positions, foot, heightmaps, x_pos, y_pos, est_robot_base_height, env_idx = get_observation(config['num_x_test'], 
                                    config['num_y_test'], envs, heightmap_params=config['heightmap_params'], vis=vis)


    loss = Loss(envs, config['device'])
    loss.to(config['device'])
    del envs

    x_pos = torch.from_numpy(x_pos).type(torch.float32).to(config['device'])
    y_pos = torch.from_numpy(y_pos).type(torch.float32).to(config['device'])
    est_robot_base_height = torch.from_numpy(est_robot_base_height).type(torch.float32).to(config['device'])
    env_idx = torch.from_numpy(env_idx).type(torch.float32).to(config['device'])

    # initialize neural network
    foot_positions = torch.from_numpy(foot_positions).type(torch.float32).to(config['device'])
    foot = torch.from_numpy(foot).unsqueeze(1).type(torch.float32).to(config['device'])
    heightmaps = torch.from_numpy(heightmaps).unsqueeze(1).type(torch.float32).to(config['device']) # add channel dimension of 1

    pred_next_step = agent(foot_positions, foot, heightmaps)

    _, info = loss.loss(pred_next_step, foot_positions, foot, x_pos, y_pos, est_robot_base_height, env_idx,
                        terrain_loss_coefficient=config['terrain_loss_coefficient'], 
                        distance_loss_coefficient=config['distance_loss_coefficient'],
                        height_loss_coefficient=config['height_loss_coefficient'])    
    
    return info





    # # fwd pass
    # foot_positions = torch.from_numpy(foot_positions).type(torch.float32)
    # foot = torch.from_numpy(foot).unsqueeze(1).type(torch.float32)
    # heightmaps = torch.from_numpy(heightmaps).unsqueeze(1).type(torch.float32) # add channel dimension of 1
    # output = agent(foot_positions, foot, heightmaps)
    # output = output.detach().numpy()
    # if vis:
    #     pred_foot_shp = envs[0].client.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=[1., 1., 1., 1.])
    #     for i in range(len(output)):
    #         basePosition = [output[i,0] + x_pos[i], output[i,1] + y_pos[i], output[i,2] + est_robot_base_height[i]]
    #         envs[0].client.createMultiBody(baseVisualShapeIndex=pred_foot_shp, basePosition=basePosition)
    #     time.sleep(1e5)



if __name__ == '__main__':
    eval_agent(None, vis=True)


