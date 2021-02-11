import numpy as np
import pybullet as p
import time
import warnings
import time
import cv2
def create_costmap(fake_client, terrain_bounds, terrain_max_height=100, mesh_res=100):
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

    # conserve memory, this array can be quite large
    rays = np.zeros((num_x * num_y, 4)).astype('float16') 
    rays[:, 0] = (np.tile(np.arange(num_x), num_y) / mesh_res + x_lb).astype('float16') 
    rays[:, 1] = (np.arange(num_y).repeat(num_x) / mesh_res + y_lb).astype('float16')
    rays[:, 2] = np.ones(num_x * num_y) * terrain_max_height
    rays[:, 3] = np.ones(num_x * num_y) * -0.1
    raw = fake_client.rayTestBatch(rayFromPositions=rays[:,[0,1,2]], rayToPositions=rays[:,[0,1,3]])
    heights = np.array([raw[i][3][2] for i in range(num_x * num_y)]).reshape((num_y, num_x))


    heights_img = np.interp(heights, [heights.min(), heights.max()], [0, 1])
    blur = cv2.GaussianBlur(heights_img,(5,5),0)
    laplacian = cv2.Laplacian(heights_img, cv2.CV_64F)
    laplacian_blur = cv2.Laplacian(blur, cv2.CV_64F)
    blur_laplacian = cv2.GaussianBlur(laplacian,(5,5),0)
    # sobelx = cv2.Sobel(heights_img,cv2.CV_64F,1,0,ksize=5)
    # sobely = cv2.Sobel(heights_img,cv2.CV_64F,0,1,ksize=5)



    # breakpoint()

    # cv2.imshow('heights', heights_img)
    # cv2.imshow('filt', laplacian)
    # cv2.imshow('blur', blur)
    cv2.imshow('laplacian_blur', laplacian_blur)
    cv2.imshow('blur_laplacian', blur_laplacian)
    # cv2.imshow('sobx', sobelx)
    # cv2.imshow('soby', sobely)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

    
    


if __name__ == '__main__':
    import gym

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
    y_lb = -0.75
    y_ub = 0.75
    create_costmap(env.fake_client, [x_lb, x_ub, y_lb, y_ub], mesh_res=50)


