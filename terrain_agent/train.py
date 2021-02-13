import torch
import numpy as np
import pybullet as p
import gym
import time
import os 
import matplotlib.pyplot as plt


N_ENVS = 1
X_SPAN = 0.483 # distance between quadruped's front and back nominal foot placements
Y_SPAN = 0.234
STEP_LEN = 0.2

def rand_foosteps(x_pos, y_pos, foot=None): #TODO vectorize this function
    if foot is None:
        foot = np.random.randint(4) # FR, FL, RR, RL
    print('#' * 100)
    print(foot)
    x = X_SPAN/2.0
    y = Y_SPAN/2.0

    pos = np.array([[x, -y, 0.], # TODO need to do a ray batch to check for the z heights of the mf. 
                    [x, y, 0.],
                    [-x, -y, 0.],
                    [-x, y, 0.]])

    if foot == 0: # move RR fwd one step len, possibly move left side fwd one half step
        pos[2, 0] += STEP_LEN 
        pos[[1, 3], 0] += np.random.uniform(0, 0.75 * STEP_LEN)
    elif foot == 1: # move RL fwd one step len, possibly move right side fwd one half step
        pos[3, 0] += STEP_LEN
        pos[[0, 2], 0] += np.random.uniform(0, 0.75 * STEP_LEN)
    elif foot == 2: # possibly move left side fwd one half step
        pos[[1, 3], 0] += np.random.uniform(0, 0.75 * STEP_LEN)
    elif foot == 3: # possibly move right side fwd one half step
        pos[[0, 2], 0] += np.random.uniform(0, 0.75 * STEP_LEN)
    else:
        assert False
    
    pos += np.random.randn(*pos.shape) * 0.0125
    pos[:, 0] += x_pos
    pos[:, 1] += y_pos
    return pos, foot

# create several envs of varying difficulty
envs = [''] * N_ENVS

for i in range(len(envs)):
    envs[i] = gym.make('gym_aliengo:AliengoSteps-v0', 
                    rows_per_m=np.random.uniform(1.0, 5.0), 
                    terrain_height_range=np.random.uniform(0, 0.375))
    envs[i].reset()
assert envs[i].terrain_length == 20  
assert envs[i].terrain_width == 10 

# """Find X_SPAN and Y_SPAN"""
# # generate quadruped footstep locations, at a grid of points
# envs[0].quadruped.reset_joint_positions(stochastic=False)
# global_pos = np.array([i[0] for i in envs[0].client.getLinkStates(envs[0].quadruped.quadruped, envs[0].quadruped.foot_links)])

# print('\nx_span: {:.3f}'.format((global_pos[[0,1],0] - global_pos[[2,3],0]).mean()))
# print('y_span: {:.3f}'.format(- (global_pos[[0,2],1] - global_pos[[1,3],1]).mean()))
# print(global_pos)

# """Plot generated footsteps. Red is the nominal footsteps when standing still, black is generated footsteps."""
# pos = rand_foosteps(0, 0)
# for i in range(4):
#     plt.plot(pos[i, 0], pos[i, 1], 'ko')


# x = X_SPAN/2.0
# y = Y_SPAN/2.0
# pos = np.array([[x, -y, 0.], # TODO need to do a ray batch to check for the z heights of the mf. 
#                 [x, y, 0.],
#                 [-x, -y, 0.],
#                 [-x, y, 0.]])
# for i in range(4):
#     plt.plot(pos[i, 0], pos[i, 1], 'r*')
# plt.grid()
# plt.show()








