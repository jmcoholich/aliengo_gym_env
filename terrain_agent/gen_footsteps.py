import torch
import numpy as np
import pybullet as p
import gym
import time
import os 
import matplotlib.pyplot as plt

def rand_foosteps(x_pos, y_pos, envs, foot=None, rayFromZ=100.0): 
    if not (isinstance(x_pos, np.ndarray) and isinstance(y_pos, np.ndarray)): 
        raise TypeError("Inputs must be np arrays.")
    if not (len(x_pos.shape) == 1 and len(y_pos.shape) == 1):
        raise ValueError('x_pos and y_pos must both be 1d np arrays.')
    # if not x_pos.shape == y_pos.shape:
    #     raise ValueError("x_pos and y_pos must have the same shape")
    # if not (len(x_pos.shape) == 2 and x_pos.shape[1] == 1):
    #     raise ValueError("Inputs must be an array with shape (n, 1) where n is batch size.")
    len_x = len(x_pos)
    len_y = len(y_pos)
    x_pos = np.expand_dims(x_pos.repeat(len_y), 1)
    y_pos = np.expand_dims(np.tile(y_pos, len_x), 1)
    assert x_pos.shape == y_pos.shape
    num_envs = len(envs)
    x_pos = np.tile(x_pos, (num_envs, 1))
    y_pos = np.tile(y_pos, (num_envs, 1))
    assert x_pos.shape == y_pos.shape
    n = x_pos.shape[0] # this is the batch size
    print('Generating {} footstep placements ({} envs * {} x-positions * {} y-positions)'.\
                                                                                    format(n, num_envs, len_x, len_y))

    if foot is None:
        foot = np.random.randint(low=0, high=4, size=(n,)) # FR, FL, RR, RL
    x = X_SPAN/2.0
    y = Y_SPAN/2.0

    pos = np.array([[x, -y, 0.],
                    [x, y, 0.],
                    [-x, -y, 0.],
                    [-x, y, 0.]])

    output = np.tile(pos, (n, 1, 1))
    # step_rand = np.random.uniform(0, 0.75 * STEP_LEN, size=(n, 1))
    # total_rand = np.random.standard_normal(output.shape) * 0.0125

    # breakpoint()
    idx = np.asarray(foot == 0).nonzero()[0]
    output[idx, 2, 0] += STEP_LEN
    output[idx[:, np.newaxis], [1, 3], 0] += np.random.uniform(0, 0.75 * STEP_LEN, size=(len(idx), 1))

    idx = np.asarray(foot == 1).nonzero()[0]
    output[idx, 3, 0] += STEP_LEN
    output[idx[:, np.newaxis], [0, 2], 0] += np.random.uniform(0, 0.75 * STEP_LEN, size=(len(idx), 1))

    idx = np.asarray(foot == 2).nonzero()[0]
    output[idx[:, np.newaxis], [1, 3], 0] += np.random.uniform(0, 0.75 * STEP_LEN, size=(len(idx), 1))

    idx = np.asarray(foot == 3).nonzero()[0]
    output[idx[:, np.newaxis], [0, 2], 0] += np.random.uniform(0, 0.75 * STEP_LEN, size=(len(idx), 1))

    output[:, :, :-1] += np.random.standard_normal((n, 4, 2)) * 0.0125
    output[:, :, 0] += x_pos
    output[:, :, 1] += y_pos

    # if foot == 0: # move RR fwd one step len, possibly move left side fwd one half step
    #     pos[2, 0] += STEP_LEN 
    #     pos[[1, 3], 0] += np.random.uniform(0, 0.75 * STEP_LEN)
    # elif foot == 1: # move RL fwd one step len, possibly move right side fwd one half step
    #     pos[3, 0] += STEP_LEN
    #     pos[[0, 2], 0] += np.random.uniform(0, 0.75 * STEP_LEN)
    # elif foot == 2: # possibly move left side fwd one half step
    #     pos[[1, 3], 0] += np.random.uniform(0, 0.75 * STEP_LEN)
    # elif foot == 3: # possibly move right side fwd one half step
    #     pos[[0, 2], 0] += np.random.uniform(0, 0.75 * STEP_LEN)
    # else:
    #     assert False
    
    # pos[:, 0] += x_pos
    # pos[:, 1] += y_pos

    # loop through envs and do rayTestBatch to find z position for every num_x * num_y points 
    for i in range(num_envs): 
        start = len_x * len_y * i 
        end = len_x * len_y * (i + 1) 
        rayFromPositions = output[start:end].copy()
        rayFromPositions[:, :, 2] = rayFromZ
        rayToPositions = rayFromPositions.copy()        
        rayToPositions[:, :, 2] = -1.0
        raw = envs[i].client.rayTestBatch(rayFromPositions=rayFromPositions.reshape(len_x * len_y * 4, 3), 
                                            rayToPositions=rayToPositions.reshape(len_x * len_y * 4, 3))
        assert len(raw) == len_x * len_y * 4
        output[start:end, :, 2] += np.array([raw[i][3][2] for i in range(len(raw))]).reshape((len_x * len_y, 4))
    # TODO visualize the heights
    return output, foot


if __name__ == '__main__':
    pass