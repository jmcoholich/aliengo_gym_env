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
X_SPAN = 0.483 # distance between quadruped's front and back nominal foot placements
Y_SPAN = 0.234 # distance between quadruped's left and right nominal foot placements
STEP_LEN = 0.2 
# np.random.seed(1)

#TODO test footstep generation, visualize it in pybullet

def main():

    # create several envs of varying difficulty
    envs = [''] * N_ENVS

    for i in range(len(envs)):
        envs[i] = gym.make('gym_aliengo:AliengoSteps-v0', 
                        rows_per_m=np.random.uniform(1.0, 5.0), 
                        terrain_height_range=np.random.uniform(0, 0.375), render=False)
    assert envs[i].terrain_length == 20  
    assert envs[i].terrain_width == 10 

    # """Find X_SPAN and Y_SPAN"""
    # # generate quadruped footstep locations, at a grid of points
    # envs[0].quadruped.reset_joint_positions(stochastic=False)
    # global_pos = np.array([i[0] for i in envs[0].client.getLinkStates(envs[0].quadruped.quadruped, envs[0].quadruped.foot_links)])

    # print('\nx_span: {:.3f}'.format((global_pos[[0,1],0] - global_pos[[2,3],0]).mean()))
    # print('y_span: {:.3f}'.format(- (global_pos[[0,2],1] - global_pos[[1,3],1]).mean()))
    # print(global_pos)

    # x = np.expand_dims(np.arange(5), 1)
    x_positions = np.linspace(0.0, envs[0].terrain_length - 1.0, NUM_X)
    y_positions = np.linspace(-envs[0].terrain_width/2.0 + 0.5, envs[0].terrain_width/2.0 - 0.5, NUM_Y)

    output, foot = rand_foosteps(x_positions, y_positions, envs)


    # """Plot generated footsteps. Red is the nominal footsteps when standing still, black is generated footsteps."""
    # idx = 0
    # pos = output[idx]
    # print('\n', foot[idx])
    # for i in range(4):
    #     plt.plot(pos[i, 0], pos[i, 1], 'ko')


    # x = X_SPAN/2.0
    # y = Y_SPAN/2.0
    # pos = np.array([[x, -y, 0.], 
    #                 [x, y, 0.],
    #                 [-x, -y, 0.],
    #                 [-x, y, 0.]])
    # for i in range(4):
    #     plt.plot(pos[i, 0], pos[i, 1], 'r*')
    # plt.grid()
    # plt.show()

if __name__ == '__main__':
    main()












