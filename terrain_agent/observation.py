import torch
import numpy as np
import pybullet as p
import gym
import time
import os 
import matplotlib.pyplot as plt

def get_observation(num_x, num_y, envs, foot=None, rayFromZ=100.0, x_span=0.483, y_span=0.234, step_len=0.2, vis=False,
                    heightmap_params=None): 
    if not (isinstance(num_x, int) and isinstance(num_y, int)): raise TypeError("num_x and num_y must be integers")
    if not (num_x > 0 and num_y > 0): raise ValueError("num_x and num_y must be positive")

    x_pos = np.linspace(0.0, envs[0].terrain_length - 1.0, num_x)
    y_pos = np.linspace(-envs[0].terrain_width/2.0 + 0.5, envs[0].terrain_width/2.0 - 0.5, num_y)
    num_envs = len(envs)
    x_pos = np.tile(np.expand_dims(x_pos.repeat(num_y), 1), (num_envs, 1))
    y_pos = np.tile(np.expand_dims(np.tile(y_pos, num_x), 1), (num_envs, 1))
    assert x_pos.shape == y_pos.shape
    n = x_pos.shape[0] # this is the batch size
    assert n == num_x * num_y * num_envs
    print('\n\nGenerating {} footstep placements ({} envs * {} x-positions * {} y-positions)'.\
                                                                                    format(n, num_envs, num_x, num_y))

    # Generate foot placements##########################################################################################
    if foot is None:
        foot = np.random.randint(low=0, high=4, size=(n,)) # FR, FL, RR, RL
    x = x_span/2.0
    y = y_span/2.0

    pos = np.array([[x, -y, 0.],
                    [x, y, 0.],
                    [-x, -y, 0.],
                    [-x, y, 0.]])

    output = np.tile(pos, (n, 1, 1))

    idx = np.asarray(foot == 0).nonzero()[0]
    output[idx, 2, 0] += step_len
    output[idx[:, np.newaxis], [1, 3], 0] += np.random.uniform(0, 0.75 * step_len, size=(len(idx), 1))

    idx = np.asarray(foot == 1).nonzero()[0]
    output[idx, 3, 0] += step_len
    output[idx[:, np.newaxis], [0, 2], 0] += np.random.uniform(0, 0.75 * step_len, size=(len(idx), 1))

    idx = np.asarray(foot == 2).nonzero()[0]
    output[idx[:, np.newaxis], [1, 3], 0] += np.random.uniform(0, 0.75 * step_len, size=(len(idx), 1))

    idx = np.asarray(foot == 3).nonzero()[0]
    output[idx[:, np.newaxis], [0, 2], 0] += np.random.uniform(0, 0.75 * step_len, size=(len(idx), 1))

    output[:, :, :-1] += np.random.standard_normal((n, 4, 2)) * 0.0125
    output[:, :, 0] += x_pos
    output[:, :, 1] += y_pos


    # loop through envs and do rayTestBatch to find z position for every num_x * num_y points 
    for i in range(num_envs): 
        start = num_x * num_y * i 
        end = num_x * num_y * (i + 1) 
        rayFromPositions = output[start:end].copy()
        rayFromPositions[:, :, 2] = rayFromZ
        rayToPositions = rayFromPositions.copy()        
        rayToPositions[:, :, 2] = -1.0
        raw = envs[i].fake_client.rayTestBatch(rayFromPositions=rayFromPositions.reshape(num_x * num_y * 4, 3), 
                                            rayToPositions=rayToPositions.reshape(num_x * num_y * 4, 3),
                                            numThreads=0)
        assert len(raw) == num_x * num_y * 4
        output[start:end, :, 2] += np.array([raw[i][3][2] for i in range(len(raw))]).reshape((num_x * num_y, 4))

    # Generate heightmap################################################################################################

    # get the heightmap around each x and y position. Heighmaps will go through CNN encoder, so store as 2D arrays
    if heightmap_params is None:
        heightmap_params = {'length': 1.25, # assumes square #TODO allow rectangular
                            'robot_position': 0.5, # distance of robot base origin from back edge of height map
                            'grid_spacing': 0.0625}
    # if vis: heightmap_params['grid_spacing'] = 0.25
    pts_per_env = num_x * num_y
    grid_len = int(heightmap_params['length']/heightmap_params['grid_spacing']) + 1
    assert pts_per_env * num_envs == len(output)
    heightmaps = np.zeros((n, grid_len, grid_len))

    # this is the estimated height of torso of quadruped. Its the average of the four feet height plus a constant
    est_robot_base_height = output[:,:,-1].mean(axis=1) + 0.48
    assert est_robot_base_height.shape[0] == n
    for i in range(num_envs):
        for j in range(pts_per_env * i, pts_per_env * (i + 1)):
            # heightmap is relative to robot height
            heightmaps[j] = envs[i].quadruped._get_heightmap(envs[i].fake_client, 
                                                        ray_start_height=rayFromZ, 
                                                        base_position=[x_pos[j], y_pos[j], est_robot_base_height[j]],
                                                        heightmap_params=heightmap_params,
                                                        vis=vis,
                                                        vis_client=envs[i].client)

    # visualization ####################################################################################################
    if vis: # just create visual shapes in all envs. Doesn't take much time in non-rendered ones anyways.
        # visualize foot position locations
        assert len(output)%num_envs == 0
        n = int(len(output)/num_envs)
        for i in range(num_envs):
            foot_shp = envs[i].client.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=[1., 0., 0., 1.])
            curr_foot_shp = envs[i].client.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=[0., 0., 1., 1.])
            torso_shp = envs[i].client.createVisualShape(p.GEOM_SPHERE,
                                                                    radius=0.04, rgbaColor=[0., 1., 0., 1.])
            for j in range(i * n, (i+1) * n):
                envs[i].client.createMultiBody(baseVisualShapeIndex=torso_shp, 
                                                        basePosition=[x_pos[j], y_pos[j], est_robot_base_height[j]])
                for k in range(4):
                    if k == foot[j]:
                        envs[i].client.createMultiBody(baseVisualShapeIndex=curr_foot_shp, basePosition=output[j,k])
                    else:
                        envs[i].client.createMultiBody(baseVisualShapeIndex=foot_shp, basePosition=output[j,k])
    output[:,:,0] -= x_pos
    output[:,:,1] -= y_pos
    est_robot_base_height = np.expand_dims(est_robot_base_height, 1)
    output[:,:,2] -= est_robot_base_height
    env_idx = np.expand_dims(np.arange(num_envs, dtype=np.int8).repeat(num_x * num_y), 1)
    return output, foot, heightmaps, x_pos, y_pos, est_robot_base_height, env_idx 


if __name__ == '__main__':
    N_ENVS = 2
    NUM_X = 2 # number of footstep placements along x direction, per env
    NUM_Y = 2 # number of footstep placements along y direction, per env
    env_to_render = 1 # can only render one env at a time because of the way pybullet works

    # create several envs of varying difficulty
    envs = [''] * N_ENVS

    for i in range(len(envs)):
        print('\n' * 2)
        envs[i] = gym.make('gym_aliengo:AliengoSteps-v0', 
                        rows_per_m=np.random.uniform(1.0, 1.5), 
                        terrain_height_range=np.random.uniform(0.25, 0.375), render=(i==env_to_render),
                        fixed=True,
                        fixed_position=[-10.0, 0.0, 1.0],
                        terrain_width=5.0,
                        terrain_length=5.0) 
                        # with multiple envs?

    output, foot, heightmaps, _, _ = get_observation(NUM_X, NUM_Y, envs, vis=True)

    print('\nDONE')
    time.sleep(1e5)

    # np.random.seed(1)


    # """Find X_SPAN and Y_SPAN"""
    # # generate quadruped footstep locations, at a grid of points
    # envs[0].quadruped.reset_joint_positions(stochastic=False)
    # global_pos = np.array([i[0] for i in envs[0].client.getLinkStates(envs[0].quadruped.quadruped, envs[0].quadruped.foot_links)])

    # print('\nx_span: {:.3f}'.format((global_pos[[0,1],0] - global_pos[[2,3],0]).mean()))
    # print('y_span: {:.3f}'.format(- (global_pos[[0,2],1] - global_pos[[1,3],1]).mean()))
    # print(global_pos)

    # x = np.expand_dims(np.arange(5), 1)



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