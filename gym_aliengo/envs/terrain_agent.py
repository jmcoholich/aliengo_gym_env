import numpy as np
import pybullet as p
import time
import warnings
import time
import cv2
import matplotlib.pyplot as plt


def norm(img):
    """Maps img back to [0,1]"""
    return np.interp(img, [img.min(), img.max()], [0, 1])


def preprocess(img, ratio):
    img = norm(img)
    width = int(img.shape[1] * ratio)
    height = int(img.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def show(name, img, ratio=4):
    cv2.imshow(name, preprocess(img, ratio))


def show_heat(name, img, ratio=4):
    img = preprocess(img, ratio)
    colormap = plt.get_cmap('inferno')
    heatmap = (colormap(img) * 2**16).astype(np.uint16)[:,:,:3]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, heatmap)


def create_costmap(fake_client, terrain_bounds, foot_pos, terrain_max_height=100, mesh_res=100):
    """
    terrain_bounds should be [x_lb, x_ub, y_lb, y_ub]. Assumes bounds are rectangular.
    mesh_res is points per m
    foot_pos is the position of the current foot on the terrain 
    """

    x_lb, x_ub, y_lb, y_ub = terrain_bounds
    x_len = x_ub - x_lb
    y_len = y_ub - y_lb
    num_x = int(x_len * mesh_res + 1)
    num_y = int(y_len * mesh_res + 1)
    foot_x, foot_y = foot_pos

    # conserve memory, this array can be quite large
    rays = np.zeros((num_x * num_y, 4)).astype('float16') 
    rays[:, 0] = (np.tile(np.arange(num_x), num_y) / mesh_res + x_lb).astype('float16') 
    rays[:, 1] = (np.arange(num_y).repeat(num_x) / mesh_res + y_lb).astype('float16')
    rays[:, 2] = np.ones(num_x * num_y) * terrain_max_height
    rays[:, 3] = np.ones(num_x * num_y) * -0.1
    raw = fake_client.rayTestBatch(rayFromPositions=rays[:,[0,1,2]], rayToPositions=rays[:,[0,1,3]])
    heights = np.array([raw[i][3][2] for i in range(num_x * num_y)]).reshape((num_y, num_x))

    heights_img = norm(heights)
    laplacian = cv2.Laplacian(heights_img, cv2.CV_64F, ksize=3)
    abs_laplacian = np.abs(laplacian)

    blur_abs_laplacian = cv2.GaussianBlur(abs_laplacian, ksize=(31,31), sigmaX=3)

    foot_filter = np.zeros((num_y, num_x))
    foot_i = int(num_y * (y_len - foot_y + y_lb)/y_len) # i indexes along the rows, which is the y axis of the terrain
    foot_j = int(num_x * (foot_x - x_lb)/x_len)
    foot_filter[foot_i, foot_j] = 1.0
    step_len = 0.2 # I want cost to be high, the higher we are from the step len
    step_i = foot_i
    step_j = int(num_x * ((foot_x + step_len) - x_lb)/x_len)
    foot_filter[step_i, step_j] = 1.0



    show('foot_filter', foot_filter)

    # breakpoint()
    show('heights', heights_img)
    # show('blur_abs_laplace', blur_abs_laplacian)
    show_heat('blur_abs_laplace', blur_abs_laplacian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  


if __name__ == '__main__':
    import gym
    np.random.seed(1)
    env = gym.make('gym_aliengo:AliengoSteps-v0', render=False)
    # env = gym.make('gym_aliengo:AliengoHills-v0', render=True)
    
    # it is too much computation to do it over the entire terrain
    # x_lb = -2.0
    # x_ub = env.terrain_length + 1.0 
    # y_lb = -env.terrain_width/2.0
    # y_ub =  env.terrain_width/2.0


    # x_lb = 5.0
    # x_ub = 5.5 
    # y_lb = -0.25
    # y_ub = 0.25
    # create_costmap(env.fake_client, [x_lb, x_ub, y_lb, y_ub])

    x_lb = 4.5
    x_ub = 6.0
    y_lb = -0.5
    y_ub = 0.5

    foot_x = 5.0
    foot_y = 0.25
    create_costmap(env.fake_client, [x_lb, x_ub, y_lb, y_ub], [foot_x, foot_y], mesh_res=50)


