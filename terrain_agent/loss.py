import numpy as np
import pybullet as p
import time
import warnings
import time
import cv2
import matplotlib.pyplot as plt
import sys
import torch
from math import floor, ceil

class Loss:
    def __init__(self, envs, device, mesh_res=50, rayFromZ=100.): #TODO up mesh_res to 100, but need to change the loss params
        # generate and store heightmap of all envs
        self.mesh_res = mesh_res

        extra_padding = 2.0 # in meters
        self.x_lb = -2.0 - extra_padding
        self.x_ub = envs[0].terrain_length + 2.0 + extra_padding
        self.y_lb = -envs[0].terrain_width/2.0 - 0.5 - extra_padding
        self.y_ub = envs[0].terrain_width/2.0 + 0.5 + extra_padding
        self.num_x = int((self.x_ub - self.x_lb) * self.mesh_res + 1) # per env
        self.num_y = int((self.y_ub - self.y_lb) * self.mesh_res + 1) # per env

        n_envs = len(envs)
        self.heightmaps = torch.zeros((n_envs, self.num_x, self.num_y))
        for i in range(n_envs): # this can't be vectorized due to use of pybullet.getRayTestBatch()
            self.heightmaps[i] = torch.from_numpy(self.get_heightmap(envs[i], rayFromZ))
        self.heightmaps = self.heightmaps.to(device)
        self.costmaps = self.create_costmaps(self.heightmaps, device)
        

    def create_costmaps(self, heightmaps, device, vis=False):
        """
        Returns a costmap tensor, calculated from a heightmap tensor
        Current involves convolution with gaussian blur and Laplacian filter. 
        Zero pad the image (presumably, the heightmap covers the edges of the terrian, beyond which everything is zero
        elevation anyways.)
        """
        assert self.mesh_res == 50 #TODO generalize gaussian filter params to different mesh resolutions 
        costmaps = heightmaps.unsqueeze(1).detach().clone() # add channel dimension. I think detach() is superfluous
        if vis: 
            pch_i = [costmaps.shape[2]//2, costmaps.shape[3]//2] # patch indices
            pch_s = 75
            show('raw heightmap', costmaps[0, 0], ratio=0.75)
            show('raw heightmap patch', 
                    costmaps[0, 0, pch_i[0]:pch_i[0] + pch_s, pch_i[1]:pch_i[1] + pch_s], 
                    ratio=4)
        g_ksize = 31
        l_ksize = 3
        g_filt = self.create_2d_gaussian(ksize=g_ksize, sigma=3.0, vis=vis).unsqueeze(0).unsqueeze(0).to(device)
        lap_filt = self.create_laplacian(ksize=l_ksize).unsqueeze(0).unsqueeze(0).to(device)
        costmaps = torch.nn.functional.conv2d(costmaps, lap_filt, padding=int((l_ksize - 1)/2)).abs()
        if vis: 
            show('lap_filtered', costmaps[0, 0], ratio=0.75)
            show('lap_filtered patch', 
                costmaps[0, 0, pch_i[0]:pch_i[0] + pch_s, pch_i[1]:pch_i[1] + pch_s], 
                ratio=4.)

        costmaps = torch.nn.functional.conv2d(costmaps, g_filt, padding=int((g_ksize - 1)/2))
        if vis: 
            show_heat('final_costmap', costmaps[0, 0], ratio=0.75)
            show_heat('final_costmap patch', 
                        costmaps[0, 0, pch_i[0]:pch_i[0] + pch_s, pch_i[1]:pch_i[1] + pch_s], 
                        ratio=4.)
        
        costmaps = costmaps.squeeze(1)
        
        if vis: 
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return costmaps


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

    
    def get_heightmap(self, env, rayFromZ, vis=False):
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


    def get_three_points(self, predicted_pos, env_idx, map_type): #TODO make env_idx a long tensor from the beginning
        """
        Takes an xyz position (z position not used though) and outputs three xyz positions (where z is the value of 
        the costmap).
        """
        if map_type == 'height':
            map_ = self.heightmaps
        elif map_type == 'cost':
            map_ = self.costmaps
        else:
            raise ValueError('incorrect map type given')
        n = predicted_pos.shape[0]
        idx_space_x = (predicted_pos[:, 0].detach() - self.x_lb) * self.mesh_res # this is shape (n,)
        idx_space_y = (predicted_pos[:, 1].detach() - self.y_lb) * self.mesh_res # this is shape (n,)

        x_floor = torch.floor(idx_space_x)
        x_ceil  = torch.ceil(idx_space_x)
        y_floor = torch.floor(idx_space_y)
        y_ceil  = torch.ceil(idx_space_y)

        # following statements prevent floor(idx) and ceil(idx) from both returning idx, giving 3 points on a line.
        # Its rare that these conditions will be True, but if this goes uncorrected, I will get NaNs.
        idx_space_x[x_floor == x_ceil] += 0.1
        idx_space_y[y_floor == y_ceil] += 0.1

        # must recalculate these
        x_floor = torch.floor(idx_space_x)
        x_ceil  = torch.ceil(idx_space_x)
        y_floor = torch.floor(idx_space_y)
        y_ceil  = torch.ceil(idx_space_y)

        output_points = torch.zeros((n, 3, 3)).to(predicted_pos.device)
        output_points[:, 0, 0] = x_floor/self.mesh_res + self.x_lb
        output_points[:, 0, 1] = y_floor/self.mesh_res + self.y_lb
        output_points[:, 0, 2] = map_[env_idx.squeeze(1).type(torch.long), x_floor.type(torch.long), y_floor.type(torch.long)]

        output_points[:, 1, 0] = x_ceil/self.mesh_res + self.x_lb
        output_points[:, 1, 1] = y_floor/self.mesh_res + self.y_lb
        output_points[:, 1, 2] = map_[env_idx.squeeze(1).type(torch.long), x_ceil.type(torch.long), y_floor.type(torch.long)]

        output_points[:, 2, 0] = x_ceil/self.mesh_res + self.x_lb
        output_points[:, 2, 1] = y_ceil/self.mesh_res + self.y_lb
        output_points[:, 2, 2] = map_[env_idx.squeeze(1).type(torch.long), x_ceil.type(torch.long), y_ceil.type(torch.long)]
        return output_points
        

    def get_plane(self, pts):
        # I only check for the case where they are along a line parallel to the xz or yz plane. 
        # if (pts[:, 0, 0] == pts[:, 1, 0] == pts[:, 2, 0]).any() or (pts[:, 0, 1] == pts[:, 1, 1] == pts[:, 2, 1]).any():
        #     raise ValueError('points are in a line') #TODO
        v1 = pts[:, 1] - pts[:, 0] # shape is (n, 3)
        v2 = pts[:, 2] - pts[:, 0] # shape is (n, 3)
        n = torch.cross(v1, v2, dim=1) # shape is (n, 3)
        d = (n * pts[:, 0]).sum(axis=1)
        coeffs = torch.stack((-n[:, 0]/n[:, 2], -n[:, 1]/n[:, 2], d/n[:, 2])).T
        return coeffs


    def loss(self, 
            pred_next_step, 
            foot_positions_, 
            foot_, 
            x_pos_,
            y_pos_, 
            est_robot_base_height_, 
            env_idx_, 
            step_len=0.2,
            distance_loss_coefficient=1.0,
            terrain_loss_coefficient=1.0 ,
            height_loss_coefficient=1.0): #TODO vectorize 
        
        foot_positions = foot_positions_.clone() 
        foot = foot_.clone()
        x_pos = x_pos_.clone()
        y_pos = y_pos_.clone() 
        est_robot_base_height = est_robot_base_height_.clone() 
        env_idx = env_idx_.clone() 

        DELTA = 0.0
        n = pred_next_step.shape[0]

        # distance_loss_array = torch.zeros((n, 1)) #TODO is predefining these necessary?
        # terrain_loss_array = torch.zeros((n, 1))
        # height_loss_array = torch.zeros((n, 1))
        
        pred_next_step[:, 0] += x_pos.squeeze() #TODO do all three additions in one step?
        pred_next_step[:, 1] += y_pos.squeeze()
        pred_next_step[:, 2] += est_robot_base_height.squeeze()

        current_pos = foot_positions[torch.arange(n), foot.squeeze().type(torch.long)]
        current_pos[:, 0] += x_pos.squeeze() #TODO do all three additions in one step?
        current_pos[:, 1] += y_pos.squeeze()
        current_pos[:, 2] += est_robot_base_height.squeeze()

        pts = self.get_three_points(pred_next_step, env_idx, map_type='cost')
        plane_coeffs = self.get_plane(pts) # x coeff, y coeff, constant term
        terrain_loss = pred_next_step[:, 0] * plane_coeffs[:, 0] + pred_next_step[:, 1] * plane_coeffs[:, 1] \
                        + plane_coeffs[:, 2]

        ideal_next_foot_pos = current_pos.clone()
        ideal_next_foot_pos[:, 0] += step_len
        distance_loss = torch.linalg.norm(ideal_next_foot_pos[:, :-1] - pred_next_step[:, :-1], ord=2, dim=1)

        pts = self.get_three_points(pred_next_step, env_idx, map_type='height')
        plane_coeffs = self.get_plane(pts) # x coeff, y coeff, constant term
        actual_height = pred_next_step[:, 0] * plane_coeffs[:, 0] + pred_next_step[:, 1] * plane_coeffs[:, 1] \
                        + plane_coeffs[:, 2]
        height_loss = (actual_height - pred_next_step[:, 2]) * (actual_height - pred_next_step[:, 2])

        if any([torch.isnan(height_loss).any(), torch.isnan(terrain_loss).any(), torch.isnan(distance_loss).any()]):
            breakpoint()


        # for i in range(n):
        #     predicted_pos = pred_next_step[i]
        #     predicted_pos[0] += x_pos[i].item()
        #     predicted_pos[1] += y_pos[i].item()
        #     predicted_pos[2] += est_robot_base_height[i].item()

        #     # get the pos of the input foot
        #     current_pos = foot_positions[i, int(foot[i])]
        #     current_pos[0] += x_pos[i].item()
        #     current_pos[1] += y_pos[i].item()
        #     current_pos[2] += est_robot_base_height[i].item()

        #     pts = self.get_three_points(predicted_pos, self.costmaps[int(env_idx[i].item())])
        #     plane_coeffs = self.get_plane(pts) # x coeff, y coeff, constant term
        #     terrain_loss = predicted_pos[0] * plane_coeffs[0] + predicted_pos[1] * plane_coeffs[1] + plane_coeffs[2]

        #     # if not (pts[:,2].min() - DELTA <= terrain_loss <= pts[:,2].max() + DELTA):
        #     #     plot_plane_interpolation(pts, plane_coeffs, terrain_loss, predicted_pos)
        #     #     breakpoint()

        #     ideal_next_foot_pos = current_pos.clone()
        #     ideal_next_foot_pos[0] += step_len
        #     distance_loss = torch.linalg.norm(ideal_next_foot_pos[:-1] - predicted_pos[:-1], ord=2)

        #     pts = self.get_three_points(predicted_pos, self.heightmaps[int(env_idx[i].item())])
        #     plane_coeffs = self.get_plane(pts) # x coeff, y coeff, constant term
        #     actual_height = predicted_pos[0] * plane_coeffs[0] + predicted_pos[1] * plane_coeffs[1] + plane_coeffs[2]
        #     height_loss = (actual_height - predicted_pos[2]) * (actual_height - predicted_pos[2])
        #     # if not (pts[:,2].min() - DELTA <= actual_height <= pts[:,2].max() + DELTA):
        #     #     plot_plane_interpolation(pts, plane_coeffs, actual_height, predicted_pos)
        #     #     breakpoint()
            
        #     if torch.isnan(terrain_loss):
        #         breakpoint()

        #     distance_loss_array[i] = distance_loss
        #     terrain_loss_array[i] = terrain_loss
        #     height_loss_array[i] = height_loss


        distance_loss_mean = distance_loss.mean()
        terrain_loss_mean = terrain_loss.mean()
        height_loss_mean = height_loss.mean()
        
        loss = distance_loss_coefficient * distance_loss_mean \
                + terrain_loss_coefficient * terrain_loss_mean \
                + height_loss_coefficient * height_loss_mean
        info = {'distance_loss': distance_loss_mean.item(),
                'terrain_loss': terrain_loss_mean.item(),
                'height_loss': height_loss_mean.item()}
        return loss, info


    def to(self, device):
        self.heightmaps = self.heightmaps.to(device)
        self.costmaps = self.costmaps.to(device)

        # self.mesh_res.to(device)
        # self.x_lb.to(device)
        # self.x_ub.to(device)
        # self.y_lb.to(device)
        # self.y_ub.to(device)
        # self.num_x.to(device)
        # self.num_y.to(device)


def plot_plane_interpolation(pts, plane_coeffs, terrain_loss, predicted_pos):
    import matplotlib.pyplot
    
    # point  = np.array([1, 2, 3])
    # normal = np.array([1, 1, 2])


    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    # d = -point.dot(normal)
    pts             =  pts.detach().numpy()          
    plane_coeffs    =  plane_coeffs
    terrain_loss    =  terrain_loss.detach().numpy()   
    predicted_pos   =  predicted_pos.detach().numpy()
    # create x,y    
    xx, yy = np.meshgrid(np.linspace(pts[:,0].min(), pts[:,0].max(), 10),
                         np.linspace(pts[:,1].min(), pts[:,1].max(), 10))

    # calculate corresponding z
    z = plane_coeffs[0].item()* xx + plane_coeffs[1].item() * yy + plane_coeffs[2].item()
    # plot the surface
    plt3d = plt.figure().gca(projection='3d')
    plt3d.plot_surface(xx, yy, z, alpha=0.2)

    # Ensure that the next plot doesn't overwrite the first plot
    ax = plt.gca()
    # ax.hold(True)
    for i in range(3):
        ax.scatter(pts[i, 0], pts[i, 1],pts[i, 2], color='green')
    ax.scatter(predicted_pos[0], predicted_pos[1], terrain_loss, color='red')
    plt.show()


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


def create_costmap(fake_client, foot_pos, terrain_max_height=100, mesh_res=100, vis=False):
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
    plot_plane_interpolation(None, None, None)
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

