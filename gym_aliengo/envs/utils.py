import numpy as np
import pybullet as p
import time
import warnings


def create_costmap(fake_client, terrain_bounds, terrain_max_height=100, mesh_res=1):
    """
    terrain_bounds should be [x_lb, x_ub, y_lb, y_ub]. Assumes bounds are rectangular.
    mesh_res is points/m 
    """

    #TODO I don't have to precompute a whole height map. I just need to compute it around the area that the footsteps 
    # are chosen in

    x_lb, x_ub, y_lb, y_ub = terrain_bounds
    x_len = x_ub - x_lb
    y_len = y_ub - y_lb
    num_x = int(x_len * mesh_res + 1)
    num_y = int(y_len * mesh_res + 1)

    
    num_x = 1
    num_y = int(1e4 + 6.25e3)
    print(terrain_bounds)
    # conserve memory, this array can be quite large
    rays = np.zeros((num_x * num_y, 4)).astype('float16') 
    rays[:, 0] = (np.tile(np.arange(num_x), num_y) / mesh_res + x_lb).astype('float16') 
    rays[:, 1] = (np.arange(num_y).repeat(num_x) / mesh_res + y_lb).astype('float16')
    rays[:, 2] = np.ones(num_x * num_y) * terrain_max_height
    rays[:, 3] = np.ones(num_x * num_y) * -0.1
    raw = fake_client.rayTestBatch(rayFromPositions=rays[:,[0,1,2]], rayToPositions=rays[:,[0,1,3]])
    # breakpoint()
    
    


if __name__ == '__main__':
    import gym

    env = gym.make('gym_aliengo:AliengoSteps-v0')

    x_lb = -2.0
    x_ub = env.terrain_length + 1.0 
    y_lb = -env.terrain_width/2.0
    y_ub =  env.terrain_width/2.0

    create_costmap(env.fake_client, [x_lb, x_ub, y_lb, y_ub])

