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


N_ENVS = 1
NUM_X = 20 # number of footstep placements along x direction, per env
NUM_Y = 20 # number of footstep placements along y direction, per env
EPOCHS = 100
LR = 1e-3
MAX_GRAD_NORM = 2.0
np.random.seed(1)
torch.manual_seed(1)

'''
TODO add noise to the observations
TODO switch to an nn that outputs a mean and std to sample from, so that I can train this with PPO instead.
TODO add wandb
'''


def main():
    device = 'cpu'
    epochs = EPOCHS


    # Data Generation ##################################################################################################
    # create several envs of varying difficulty
    envs = [''] * N_ENVS
    for i in range(len(envs)):
        envs[i] = gym.make('gym_aliengo:AliengoSteps-v0', # NOTE changing env type will break code
                        rows_per_m=np.random.uniform(1.0, 5.0),
                        terrain_height_range=np.random.uniform(0.0, 0.375), render=False,
                        fixed=True,
                        fixed_position=[-10,0,1.0])
    assert envs[i].terrain_length == 20  
    assert envs[i].terrain_width == 10 

    heightmap_params = {'length': 1.25, # TODO allow rectangular
                        'robot_position': 0.5, 
                        'grid_spacing': 1.0/64.0}

    # generate all data in the beginning
    foot_positions, foot, heightmaps, x_pos, y_pos, est_robot_base_height, env_idx = get_observation(NUM_X, NUM_Y, envs,
                                                                                    heightmap_params=heightmap_params)


    # Training #########################################################################################################
    # initialize loss object
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
    
    # initialize optimizer 
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    # train
    for i in range(epochs):
        optimizer.zero_grad()

        foot_positions_ = foot_positions.clone()
        foot_ = foot.clone()
        heightmaps_ = heightmaps.clone()

        pred_next_step = agent(foot_positions, foot, heightmaps)

        assert (foot_positions_ == foot_positions).all()
        assert (foot_ == foot).all()
        assert (heightmaps_ == heightmaps).all()


        print()
        print('NN output mean, max, min')
        print(pred_next_step.cpu().detach().numpy().mean(axis=0))
        print(pred_next_step.cpu().detach().numpy().max(axis=0))
        print(pred_next_step.cpu().detach().numpy().min(axis=0))
        print()

        # stuff = [pred_next_step, foot_positions, foot, x_pos, y_pos, est_robot_base_height, env_idx]
        # stuff_ = [None] * len(stuff)
        # for i in range(len(stuff)):
        #     stuff_[i] = stuff[i].clone()
        
        loss_, info = loss.loss(pred_next_step, foot_positions, foot, x_pos, y_pos, est_robot_base_height, env_idx) 

        # for i in range(1, len(stuff)):
        #     if not (stuff_[i] == stuff[i]).all():
        #         breakpoint()

        print('Gradient norm is {}'.format(torch.linalg.norm(torch.nn.utils.parameters_to_vector(agent.parameters()))))
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=MAX_GRAD_NORM)
        print(info)
        loss_.backward()
        optimizer.step()
        print('#' * 100 + '\nFinished epoch {}\n'.format(i) + '#' * 100)


if __name__ == '__main__':
    main()












