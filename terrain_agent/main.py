import torch
import numpy as np
import pybullet as p
import gym
import time
import os 
import matplotlib.pyplot as plt
import time
import wandb

from observation import get_observation
from agent import TerrainAgent
from loss import Loss
from eval_agent import eval_agent


N_ENVS = 2
NUM_X = 20 # number of footstep placements along x direction, per env
NUM_Y = 20 # number of footstep placements along y direction, per env
N_ENVS_TEST = 3
NUM_X_TEST = 10
NUM_Y_TEST = 10
EPOCHS = 100
LR = 3e-5
MAX_GRAD_NORM = 2.0
WANDB_PROJECT = 'terrain_agent_pretrain'
SEED = 1
DEVICE = 'cuda'
ENV = 'gym_aliengo:AliengoSteps-v0'
TERRAIN_LOSS_COEFF = 10.0
HEIGHT_LOSS_COEFF = 1.0
DISTANCE_LOSS_COEFF = 1.0
VERBOSE = False
EVAL_INTERVAL = 5
np.random.seed(SEED)
torch.manual_seed(SEED)


'''
TODO add noise to the observations
TODO switch to an nn that outputs a mean and std to sample from, so that I can train this with PPO instead.
TODO implement minibatches
TODO generate large test set in advance
'''


def main():
    start_time = time.time()
    heightmap_params = {'length': 1.25, # TODO allow rectangular
                        'robot_position': 0.5, 
                        'grid_spacing': 1.0/64.0}
    config = {'n_envs': N_ENVS, 'num_x': NUM_X, 'num_y': NUM_Y, 'epochs': EPOCHS, 'lr': LR, 
                'max_grad_norm:': MAX_GRAD_NORM, 'seed': SEED, 'device': DEVICE, 'env': ENV, 
                'distance_loss_coefficient': DISTANCE_LOSS_COEFF, 'height_loss_coefficient': HEIGHT_LOSS_COEFF, 
                'terrain_loss_coefficient':TERRAIN_LOSS_COEFF, 'eval_interval': EVAL_INTERVAL, 
                'n_envs_test': N_ENVS_TEST, 'num_x_test':NUM_X_TEST, 'num_y_test':NUM_Y_TEST, 'env': ENV, 
                'heightmap_params':heightmap_params} 
    wandb.init(project=WANDB_PROJECT, config=config)
    device = DEVICE
    epochs = EPOCHS


    # Data Generation ##################################################################################################
    # create several envs of varying difficulty
    envs = [''] * N_ENVS
    for i in range(len(envs)):
        envs[i] = gym.make(ENV, # NOTE changing env type will break code
                        rows_per_m=np.random.uniform(1.0, 5.0),
                        terrain_height_range=np.random.uniform(0.0, 0.375), render=False,
                        fixed=True,
                        fixed_position=[-10,0,1.0])
    assert envs[i].terrain_length == 20  
    assert envs[i].terrain_width == 10 


    # generate all data in the beginning
    # TODO multithread the stuff in get_observation before I start getting the heightmaps
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
    wandb.watch(agent)
    
    # initialize optimizer 
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    # train
    for i in range(epochs):

        if VERBOSE: start = time.time()
        optimizer.zero_grad()

        a = time.time()
        pred_next_step = agent(foot_positions, foot, heightmaps)
        
        if VERBOSE:
            print()
            print('NN output mean, max, min')
            print(pred_next_step.cpu().detach().numpy().mean(axis=0))
            print(pred_next_step.cpu().detach().numpy().max(axis=0))
            print(pred_next_step.cpu().detach().numpy().min(axis=0))
            print()

        if VERBOSE: b = time.time()
        loss_, info = loss.loss(pred_next_step, foot_positions, foot, x_pos, y_pos, est_robot_base_height, env_idx,
                        terrain_loss_coefficient=TERRAIN_LOSS_COEFF, distance_loss_coefficient=DISTANCE_LOSS_COEFF,
                        height_loss_coefficient=HEIGHT_LOSS_COEFF) 
        for key in ['distance_loss', 'terrain_loss', 'height_loss']:
            info['train_' + key] = info.pop(key)

        if VERBOSE: asdf = time.time()
        if VERBOSE:
            print('Gradient norm is {}'.format(torch.linalg.norm(torch.nn.utils.parameters_to_vector(agent.parameters()))))
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=MAX_GRAD_NORM)
        if VERBOSE: print(info)
        if VERBOSE: c = time.time()
        loss_.backward()
        if VERBOSE: d = time.time()
        optimizer.step()
        if VERBOSE: print('#' * 100 + '\nFinished epoch {}\n'.format(i) + '#' * 100)
        e = time.time()
        info.update({'min_wall_time': (e - start_time)/60., 'epoch': i})
        if VERBOSE:
            print()
            print('zero grad time: {:.4f}'.format(a-start))
            print('fwd pass time: {:.4f}'.format(b-a))
            print('loss time: {:.4f}'.format(asdf-b))
            print('clip grad time: {:.4f}'.format(c-asdf))
            print('backwards pass time: {:.4f}'.format(d-c))
            print('optimizer step time: {:.4f}'.format(e-d))


        if i%5 == 0:
            test_info = eval_agent(agent, config)
            for key in ['distance_loss', 'terrain_loss', 'height_loss']:
                test_info['test_' + key] = test_info.pop(key)
            info.update(test_info)
        
        wandb.log(info)
if __name__ == '__main__':
    main()












