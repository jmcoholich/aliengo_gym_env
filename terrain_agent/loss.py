import numpy as np
import pybullet as p
import time
import warnings
import time
import cv2
import matplotlib.pyplot as plt
import sys
import torch

class Loss:
    def __init__(self, envs, device, mesh_res=50, rayFromZ=100.): #TODO up mesh_res to 100, but need to change the loss params
        # generate and store heightmap of all envs
        self.mesh_res = mesh_res
        self.x_lb = -2.0
        self.x_ub = envs[0].terrain_length + 2.0
        self.y_lb = -envs[0].terrain_width/2.0 - 0.5
        self.y_ub = envs[0].terrain_width/2.0 + 0.5
        self.num_x = int((self.x_ub - self.x_lb) * self.mesh_res + 1) # per env
        self.num_y = int((self.y_ub - self.y_lb) * self.mesh_res + 1) # per env

        n_envs = len(envs)
        self.heightmaps = torch.zeros((n_envs, self.num_x, self.num_y))
        for i in range(n_envs): # this can't be vectorized due to use of pybullet.getRayTestBatch()
            self.heightmaps[i] = torch.from_numpy(self.get_heightmap(envs[i], rayFromZ))
        self.to(device)
        self.convert_heightmaps_to_costmaps()
        

    def convert_heightmaps_to_costmaps(self, vis=False):
        """
        Converts a grid of heights to a terrain costmap.
        Current involves convolution with gaussian blur and Laplacian filter. 
        Zero pad the image (presumably, the heightmap covers the edges of the terrian, beyond which everything is zero
        elevation anyways.)
        """
        assert self.mesh_res == 50 #TODO generalize gaussian filter params to different mesh resolutions 
        self.heightmaps = self.heightmaps.unsqueeze(1)
        if vis: 
            pch_i = [self.heightmaps.shape[2]//2, self.heightmaps.shape[3]//2] # patch indices
            pch_s = 75
            show('raw heightmap', self.heightmaps[0, 0], ratio=0.75)
            show('raw heightmap patch', 
                    self.heightmaps[0, 0, pch_i[0]:pch_i[0] + pch_s, pch_i[1]:pch_i[1] + pch_s], 
                    ratio=4)
        g_ksize = 31
        l_ksize = 3
        g_filt = self.create_2d_gaussian(ksize=g_ksize, sigma=3.0, vis=vis).unsqueeze(0).unsqueeze(0)
        lap_filt = self.create_laplacian(ksize=l_ksize).unsqueeze(0).unsqueeze(0)

        self.heightmaps = torch.nn.functional.conv2d(self.heightmaps, 
                                                lap_filt, 
                                                padding=int((l_ksize - 1)/2)).abs()
        if vis: 
            show('lap_filtered', self.heightmaps[0, 0], ratio=0.75)
            show('lap_filtered patch', 
                self.heightmaps[0, 0, pch_i[0]:pch_i[0] + pch_s, pch_i[1]:pch_i[1] + pch_s], 
                ratio=4.)

        self.heightmaps = torch.nn.functional.conv2d(self.heightmaps, 
                                                    g_filt, 
                                                    padding=int((g_ksize - 1)/2))
        if vis: 
            show_heat('final_costmap', self.heightmaps[0, 0], ratio=0.75)
            show_heat('final_costmap patch', 
                        self.heightmaps[0, 0, pch_i[0]:pch_i[0] + pch_s, pch_i[1]:pch_i[1] + pch_s], 
                        ratio=4.)
        
        self.heightmaps = self.heightmaps.squeeze()
        
        if vis: 
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def create_laplacian(self, ksize):
        if ksize != 3: raise NotImplementedError('kernel size must be 3 for now')

        return torch.Tensor([   [0., 1., 0.],
                                [1., -4., 1.],
                                [0., 1., 0.]])

    def create_2d_gaussian(self, ksize, sigma, vis=False):
        if ksize%2 == 0: raise ValueError('square kernel size must be odd')
        n = (ksize - 1)/2.0
        x = torch.linspace(-n, n, ksize)
        x = torch.exp(-0.5 * (x/sigma) * (x/sigma)) # omit coefficients since I will normalize it anyways
        x = x.unsqueeze(1) @ x.unsqueeze(0)
        x /= x.sum()
        if vis: 
            import cv2
            show('gaussian filter', x.numpy())
        return x 

    
    def get_heightmap(self, env, rayFromZ, vis=True): #TODO vis
        # NOTE if the env is not AliengoSteps, this will throw an error

        max_batch_size = int(1e4 + 6e3) # this is the approx max number of rays you can do at a time with rayTestBatch
        assert env.terrain_length == 20  
        assert env.terrain_width == 10 


        num_pts = self.num_x * self.num_y
        if num_pts > 250 * 1e6: # if the ray_array will use more than a gig of memory
            raise RuntimeError('Number of points in terrain heightmap for loss is too high.'.format(num_pts))

        ray_array = np.zeros((num_pts, 4), dtype=np.float32)
        ray_array[:, 0] = np.linspace(self.x_lb, self.x_ub, self.num_x).repeat(self.num_y)
        ray_array[:, 1] = np.tile(np.linspace(self.y_lb, self.y_ub, self.num_y), self.num_x)
        ray_array[:, 2] = rayFromZ
        ray_array[:, 3] = -1.0
        
        heights = np.zeros(num_pts, dtype=np.float32)
        for i in range(int(num_pts//max_batch_size)):
            start = int(i * max_batch_size)
            end = int((i+1) * max_batch_size)
            raw = env.client.rayTestBatch(rayFromPositions=ray_array[start:end, [0,1,2]], 
                                            rayToPositions=ray_array[start:end, [0,1,3]], 
                                            numThreads=0)
            heights[start: end] = np.array([raw[j][3][2] for j in range(max_batch_size)])    
        raw = env.client.rayTestBatch(rayFromPositions=ray_array[end:, [0,1,2]], 
                                        rayToPositions=ray_array[end:, [0,1,3]], 
                                        numThreads=0)
        assert len(raw) == num_pts%max_batch_size
        heights[end:] = np.array([raw[j][3][2] for j in range(num_pts%max_batch_size)])
        heights = heights.reshape((self.num_x, self.num_y)) #NOTE verified this is the correct order of dimensions
        if vis:
            shape = env.client.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=[1., 0., 0., 1.])
            step = 100
            for i in range(0, self.num_x, step):
                for j in range(0, self.num_y, step):
                    k = j + i * self.num_y
                    env.client.createMultiBody(baseVisualShapeIndex=shape, 
                                            basePosition=[ray_array[k, 0], ray_array[k, 1], heights[i, j]])

            # for i in range(0, num_pts, 1000):
            #     env.client.createMultiBody(baseVisualShapeIndex=shape, 
            #                                 basePosition=[ray_array[i, 0], ray_array[i, 1], heights[i]])
        
        return heights


    def loss(self, pred_next_step, foot_positions, foot, x_pos, y_pos, est_robot_base_height, env_idx): #TODO vectorize
        n = pred_next_step.shape[0]
        num_passed_test = 0
        for i in range(n):
            # get the xy of the input foot
            current_xyz = foot_positions[i, int(foot[i])]
            current_xyz[0] += x_pos[i].item()
            current_xyz[1] += y_pos[i].item()
            current_xyz[2] += est_robot_base_height[i].item()

            # convert this xy position to an index of the heights array.
            x_idx = int(torch.round((current_xyz[0] - self.x_lb) * self.mesh_res))
            y_idx = int(torch.round((current_xyz[1] - self.y_lb) * self.mesh_res))
            if x_idx > self.num_x - 1 or y_idx > self.num_y - 1:
                raise ValueError('The indices of the loss heightfield are out of bounds.'
                ' Try increasing bounds of heightfield or check for bugs')

            closest_mesh_xyz = torch.Tensor(3)
            closest_mesh_xyz[0] = x_idx/self.mesh_res + self.x_lb
            closest_mesh_xyz[1] = y_idx/self.mesh_res + self.y_lb
            closest_mesh_xyz[2] = self.heightmaps[int(env_idx[i])][x_idx, y_idx]

            if abs(closest_mesh_xyz[2] - current_xyz[2]) < 1e-3: num_passed_test += 1

            # assert the closest mesh point is not more than square diag away 
            max_dist = 1.0/self.mesh_res * (2**0.5)/2.0
            dist = ((current_xyz[0] - closest_mesh_xyz[0])**2 + (current_xyz[1] - closest_mesh_xyz[1])**2)**0.5
            assert dist <= max_dist, '{} percent higher than max dist'.format((dist-max_dist)/max_dist * 100)
            # get the costmap surrounding it
        print('{} percent passes z-matching test for {} datapoints'.format(float(num_passed_test)/n), n)


    # def get_costmap(self, x_idx, y_idx, env_idx):
    #     # get 
    #     pass

        

    def to(self, device):
        self.heightmaps.to(device)




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


def create_costmap(fake_client, foot_pos, terrain_max_height=100, mesh_res=100, vis=False): #TODO pass the env idx to this as well.
    """
    terrain_bounds should be [x_lb, x_ub, y_lb, y_ub]. Assumes bounds are rectangular.
    mesh_res is points per m
    foot_pos is the position of the current foot on the terrain 
    """

    foot_x, foot_y = foot_pos
    x_lb = foot_x - 0.5
    x_ub = foot_x + 1.0
    y_lb = foot_y - 0.5
    y_ub = foot_y + 0.5
    step_len = 0.2 

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

    heights_img = norm(heights)
    laplacian = cv2.Laplacian(heights_img, cv2.CV_64F, ksize=3)
    abs_laplacian = np.abs(laplacian)

    blur_abs_laplacian = cv2.GaussianBlur(abs_laplacian, ksize=(31,31), sigmaX=3)

    if vis: foot_filter = np.zeros((num_y, num_x))
    foot_i = int(num_y * (y_len - foot_y + y_lb)/y_len) # i indexes along the rows, which is the y axis of the terrain
    foot_j = int(num_x * (foot_x - x_lb)/x_len)
    if vis: foot_filter[foot_i, foot_j] = 1.0
    step_i = foot_i
    step_j = int(num_x * ((foot_x + step_len) - x_lb)/x_len)
    if vis: foot_filter[step_i, step_j] = 1.0
    if vis: foot_placement_vis = foot_filter.copy()
    # penalize distance from step_i, step_j

    foot_filter = np.expand_dims((np.arange(num_y) - step_i) * (np.arange(num_y) - step_i), 1) + \
                    np.expand_dims((np.arange(num_x) - step_j) * (np.arange(num_x) - step_j), 0)

    if vis:
        show('foot_filter', foot_filter)
        show('heights', heights_img)
        show_heat('blur_abs_laplace', blur_abs_laplacian)
        show_heat('foot_filter', foot_filter)
        show('foot_placement_vis', foot_placement_vis)
        show_heat('combined', norm(blur_abs_laplacian) + norm(foot_filter))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import gym
    np.random.seed(1)
    env = gym.make('gym_aliengo:AliengoSteps-v0', render=False)
    # env = gym.make('gym_aliengo:AliengoHills-v0', render=False)
    
    foot_x = 5.1
    foot_y = 0.25
    start = time.time()
    create_costmap(env.fake_client, [foot_x, foot_y], mesh_res=50, vis=True)
    end = time.time()
    print(end-start)

