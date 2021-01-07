import gym
from gym import error, spaces, utils
from gym.utils import seeding
import sys
import pybullet as p
import os
import time
import numpy as np
import warnings
from cv2 import putText, FONT_HERSHEY_SIMPLEX, imwrite, cvtColor, COLOR_RGB2BGR
from gym_aliengo.envs import aliengo
from pybullet_utils import bullet_client as bc
from math import ceil

'''
Env for steps (random "grid" of elevated rectangles), meant to replicate the Steps env used in this paper: 
https://robotics.sciencemag.org/content/robotics/5/47/eabc5986.full.pdf
'''
class AliengoSteps(gym.Env):
    def __init__(self, render=False, realTime=False,
                row_width=0.5, # range from [0.1, 1.0] (hard to easy) default=0.5
                terrain_height_range=0.25): # range from [0.0, 0.375] (easy to hard) default=0.25

        # Environment Options
        self._apply_perturbations = False
        self.perturbation_rate = 0.00 # probability that a random perturbation is applied to the torso
        self.max_torque = 40.0 
        self.kp = 1.0 
        self.kd = 1.0
        self.n_hold_frames = 1 
        self._is_render = render
        self.eps_timeout = 240.0/self.n_hold_frames * 20 # number of steps to timeout after
        self.realTime = realTime

        # Terrain parameters, all units in meters
        assert row_width > 0.01
        self.row_width = row_width
        self.terrain_height_range = terrain_height_range # +/- half of this value to the height mean 
        self.terrain_length = 50
        self.terrain_width = 3 
        self.terrain_height = terrain_height_range/2. + 0.05 # this is just the mean height of the blocks

        self.block_length_range = self.row_width/2. # the mean is set to the same as row_width. 
        self.ramp_distance = self.terrain_height * 4

        # heightmap parameters
        self.length = 1.25 # assumes square 
        self.robot_position = 0.5 # distance of robot base origin from back edge of height map
        self.grid_spacing = 0.125 
        assert self.length%self.grid_spacing == 0
        self.grid_len = int(self.length/self.grid_spacing) + 1


        if self._is_render:
            self.client = bc.BulletClient(connection_mode=p.GUI)
        else:
            self.client = bc.BulletClient(connection_mode=p.DIRECT)
        self.fake_client = bc.BulletClient(connection_mode=p.DIRECT) # this is only used for getting the heightmap 

        self.client.setPhysicsEngineParameter(enableFileCaching=0) # load the newly generated terrain every reset()
        self.fake_client.setPhysicsEngineParameter(enableFileCaching=0)

        if (self.client == -1) or (self.fake_client == -1):
            raise RuntimeError('Pybullet could not connect to physics client')

        # (50) applied torque, pos, and vel for each motor, base orientation (quaternions), foot normal forces,
        # cartesian base acceleration, base angular velocity
        self.state_space_dim = 12 * 3 + 4 + 4 + 3 + 3 + self.grid_len**2
        self.num_joints = 18 # This includes fixed joints from the URDF
        self.action_space_dim = 12 # this remains unchanged

        self.state = np.zeros(self.state_space_dim) # I currently don't distinguish between state and observation
        self.applied_torques = np.zeros(12) 
        self.joint_velocities = np.zeros(12)
        self.joint_positions = np.zeros(12)
        self.base_orientation = np.zeros(4)
        self.foot_normal_forces = np.zeros(4)
        self.cartesian_base_accel = np.zeros(3) 
        self.base_twist = np.zeros(6) # used to calculate accelerations, angular vel included in state
        self.previous_base_twist = np.zeros(6) # used to calculate accelerations, angular vel included in state
        self.base_position = np.zeros(3) # not returned as observation, but used for calculating reward or termination
        self.eps_step_counter = 0 # Used for triggering timeout
        self.t = 0 # represents the actual time
        self._debug_ids = []

        self.reset()

        self.reward = 0 # this is to store most recent reward
        self.action_lb, self.action_ub = self.quadruped.get_joint_position_bounds()
        self.action_space = spaces.Box(
            low=self.action_lb,
            high=self.action_ub,
            dtype=np.float32
            )

        observation_lb, observation_ub = self._find_space_limits()
        self.observation_space = spaces.Box(
            low=observation_lb,
            high=observation_ub,
            dtype=np.float32
            )


    def _is_non_foot_ground_contact(self): #TODO if I ever use this in this env, account for stepping stone contact
        """Detect if any parts of the robot, other than the feet, are touching the ground."""

        raise NotImplementedError

        contact = False
        for i in range(self.num_joints):
            if i in self.quadruped.foot_links: # the feet themselves are allow the touch the ground
                continue
            points = self.client.getContactPoints(self.quadruped.quadruped, self.plane, linkIndexA=i)
            if len(points) != 0:
                contact = True
        return contact


    def _get_foot_contacts(self): 
        '''
        Returns a numpy array of shape (4,) containing the normal forces on each foot with any object in simulation. 
        '''

        contacts = [0] * 4
        for i in range(len(self.quadruped.foot_links)):
            info = self.client.getContactPoints(bodyA=self.quadruped.quadruped,  
                                        linkIndexA=self.quadruped.foot_links[i])
            if len(info) == 0: # leg does not contact ground
                contacts[i] = 0 
            elif len(info) == 1: # leg has one contact with ground
                contacts[i] = info[0][9] # contact normal force
            else: # use the contact point with the max normal force when there is more than one contact on a leg 
                #TODO investigate scenarios with more than one contact point and maybe do something better (mean 
                # or norm of contact forces?)
                normals = [info[i][9] for i in range(len(info))] 
                contacts[i] = max(normals)
                # print('Number of contacts on one foot: %d' %len(info))
                # print('Normal Forces: ', normals,'\n')
        contacts = np.array(contacts)
        if (contacts > 10_000).any():
            warnings.warn("Foot contact force of %.2f over 10,000 (maximum of observation space)" %max(contacts))

        return contacts 


    def step(self, action):
        if not ((self.action_lb <= action) & (action <= self.action_ub)).all():
            print("Action passed to env.step(): ", action)
            raise ValueError('Action is out-of-bounds of:\n' + str(self.action_lb) + '\nto\n' + str(self.action_ub)) 
            
        self.quadruped.set_joint_position_targets(action)

        if (np.random.rand() > self.perturbation_rate) and self._apply_perturbations: 
            raise NotImplementedError
            self._apply_perturbation()
        for _ in range(self.n_hold_frames):
            self.client.stepSimulation()
        self.eps_step_counter += 1
        self._update_state()
        done, info = self._is_state_terminal()
        self.reward = self._reward_function()

        if done:
            self.eps_step_counter = 0

        return self.state, self.reward, done, info

        
    def _apply_perturbation(self):
        raise NotImplementedError
        if np.random.rand() > 0.5: # apply force
            force = tuple(10 * (np.random.rand(3) - 0.5))
            self.client.applyExternalForce(self.quadruped, -1, force, (0,0,0), p.LINK_FRAME)
        else: # apply torque
            torque = tuple(0.5 * (np.random.rand(3) - 0.5))
            self.client.applyExternalTorque(self.quadruped, -1, torque, p.LINK_FRAME)


    def reset(self):
        '''Resets the robot to a neutral standing position, knees slightly bent. The motor control command is to 
        prevent the robot from jumping/falling on first user command. Simulation is stepped to allow robot to fall
        to ground and settle completely.'''

        self.client.resetSimulation()
        self.fake_client.resetSimulation() 
       
        self.client.setTimeStep(1/240.)
        self.client.setGravity(0,0,-9.8)
        self.client.setRealTimeSimulation(self.realTime) # this has no effect in DIRECT mode, only GUI mode
        
        self.plane = self.client.loadURDF(os.path.join(os.path.dirname(__file__), '../urdf/plane.urdf'))
        self.fake_plane = self.fake_client.loadURDF(os.path.join(os.path.dirname(__file__), '../urdf/plane.urdf'))
        self._create_steps()
        self.quadruped = aliengo.Aliengo(pybullet_client=self.client, 
                                        max_torque=self.max_torque, 
                                        kp=self.kp, 
                                        kd=self.kd)
        
        self.client.resetBasePositionAndOrientation(self.quadruped.quadruped,
                                            posObj=[0,0,0.48], 
                                            ornObj=[0,0,0,1.0]) 

        self.quadruped.reset_joint_positions(stochastic=True) # will put all joints at default starting positions
        for i in range(500): # to let the robot settle on the ground.
            self.client.stepSimulation()
        self._update_state()
        
        return self.state


    def _create_steps(self):
        '''Creates an identical steps terrain in client and fake client'''
        
        # pick a list of discrete values for heights to only generate a finite number of collision shapes
        n_shapes = 10
        height_values = np.linspace(-self.terrain_height_range/2., self.terrain_height_range/2., n_shapes)
        length_values = np.linspace(self.row_width - self.block_length_range/2.,
                                    self.row_width + self.block_length_range/2.,
                                    n_shapes)      
        shapeId = np.zeros(n_shapes, dtype=np.int)
        fake_shapeId = np.zeros(n_shapes, dtype=np.int)
        for i in range(len(length_values)):
            halfExtents=[length_values[i]/2., self.row_width/2., self.terrain_height/2.] 
            shapeId[i] = self.client.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents)
            fake_shapeId[i] = self.fake_client.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents)

        for row in range(int(ceil(self.terrain_width/self.row_width))):
            total_len = 0
            while total_len < self.terrain_length:
                i = np.random.randint(0, n_shapes)
                j = np.random.randint(0, n_shapes)
                if total_len < self.ramp_distance:
                    offset = self.terrain_height * (1 - float(total_len)/self.ramp_distance)
                else:
                    offset = 0
                pos = [total_len + length_values[i]/2. + 0.5, # X
                        row * self.row_width - self.terrain_width/2. + self.row_width/2., # Y
                        height_values[j] - offset + self.terrain_height/2.] # Z
                self.client.createMultiBody(baseCollisionShapeIndex=shapeId[i], basePosition=pos)
                self.fake_client.createMultiBody(baseCollisionShapeIndex=fake_shapeId[i], basePosition=pos)
                total_len += length_values[i]
      
        
     
    def _get_heightmap(self):
        '''Debug flag enables printing of labeled coordinates and measured heights to rendered simulation. 
        Uses the "fake_client" simulation instance in order to avoid robot collisions
        TODO put this into the aliengo class'''

        debug = False
        show_xy = False

        if self._debug_ids != []: # remove the exiting debug items
            for _id in self._debug_ids:
                self.client.removeUserDebugItem(_id)
            self._debug_ids = []

        base_x = self.base_position[0]
        base_y = self.base_position[1]
        base_z = self.base_position[2]


        x = np.linspace(0, self.length, self.grid_len)
        y = np.linspace(-self.length/2.0, self.length/2.0, self.grid_len)
        coordinates = np.array(np.meshgrid(x,y))
        coordinates[0,:,:] += base_x - self.robot_position
        coordinates[1,:,:] += base_y  
        # coordinates has shape (2, self.grid_len, self.grid_len)
        coor_list = coordinates.reshape((2, self.grid_len**2)).swapaxes(0, 1) # is now shape (self.grid_len**2,2) 
        ray_start = np.append(coor_list, np.ones((self.grid_len**2, 1)) * \
                                                    (self.terrain_height + self.terrain_height_range/2. + 1.), axis=1)
        ray_end = np.append(coor_list, np.zeros((self.grid_len**2, 1)) - 1, axis=1)
        raw_output = self.fake_client.rayTestBatch(ray_start, ray_end) 
        z_heights = np.array([raw_output[i][3][2] for i in range(self.grid_len**2)])
        relative_z_heights = z_heights - base_z

        if debug:
            # #print xy coordinates of robot origin 
            # _id = self.client.addUserDebugText(text='%.2f, %.2f'%(base_x, base_y),
            #             textPosition=[base_x, base_y,self.hills_height+1],
            #             textColorRGB=[0,0,0])
            # self._debug_ids.append(_id)
            for i in range(self.grid_len):
                for j in range(self.grid_len):
                    if show_xy:
                        text = '%.2f, %.2f, %.2f'%(coordinates[0,i,j], coordinates[1,i,j], z_heights.reshape((self.grid_len, self.grid_len))[i,j])
                    else:
                        text = '%.2f'%(z_heights.reshape((self.grid_len, self.grid_len))[i,j])
                    _id = self.client.addUserDebugText(text=text,
                                            textPosition=[coordinates[0,i,j], coordinates[1,i,j],self.terrain_height+0.5],
                                            textColorRGB=[0,0,0])
                    self._debug_ids.append(_id)
                    _id = self.client.addUserDebugLine([coordinates[0,i,j], coordinates[1,i,j],self.terrain_height+0.5],
                                            [coordinates[0,i,j], coordinates[1,i,j], 0],
                                            lineColorRGB=[0,0,0] )
                    self._debug_ids.append(_id)
        
        return relative_z_heights.reshape((self.grid_len, self.grid_len))
        

    def render(self, mode='human', client=None):
        '''default mode does nothing. 'rgb_array' returns image of env. '''

        if client is None: # for some reason I can't use self.client as a default value in the function definition line.
            client = self.client 

        RENDER_WIDTH = 480 
        RENDER_HEIGHT = 360

        base_x_velocity = np.array(self.client.getBaseVelocity(self.quadruped.quadruped)).flatten()[0]
        torque_pen = -0.00001 * np.power(self.applied_torques, 2).mean()

        # RENDER_WIDTH = 960 
        # RENDER_HEIGHT = 720

        # RENDER_WIDTH = 1920
        # RENDER_HEIGHT = 1080

        if mode == 'rgb_array':
            base_pos, _ = self.client.getBasePositionAndOrientation(self.quadruped.quadruped)
            # base_pos = self.minitaur.GetBasePosition()
            view_matrix = client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_pos,
                distance=2.0, 
                yaw=0,
                pitch=-30.,
                roll=0,
                upAxisIndex=2)
            proj_matrix = client.computeProjectionMatrixFOV(fov=60,
                aspect=float(RENDER_WIDTH) /
                RENDER_HEIGHT,
                nearVal=0.1,
                farVal=100.0)
            _, _, px, _, _ = client.getCameraImage(width=RENDER_WIDTH,
                height=RENDER_HEIGHT,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL)
            img = np.array(px)
            img = img[:, :, :3]
            img = putText(np.float32(img), 'X-velocity:' + str(base_x_velocity)[:6], (1, 60), 
                            FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
            img = putText(np.float32(img), 'Torque Penalty Term: ' + str(torque_pen)[:8], (1, 80), 
                            FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
            img = putText(np.float32(img), 'Total Rew: ' + str(torque_pen + base_x_velocity)[:8], (1, 100), 
                            FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
            foot_contacts = self._get_foot_contacts()
            for i in range(4):
                if type(foot_contacts[i]) is list: # multiple contacts
                    assert False
                    num = np.array(foot_contacts[i]).round(2)
                else:
                    num = round(foot_contacts[i], 2)
                img = putText(np.float32(img), 
                            ('Foot %d contacts: ' %(i+1)) + str(num), 
                            (200, 60 + 20 * i), 
                            FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
            # img = putText(np.float32(img), 
            #                 'Body Contact: ' + str(self._is_non_foot_ground_contact()), 
            #                 (200, 60 + 20 * 4), 
            #                 FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
            img = putText(np.float32(img), 
                            'Self Collision: ' + str(self.quadruped.self_collision()), 
                            (200, 60 + 20 * 5), 
                            FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
            return np.uint8(img)

        else: 
            return


    def close(self):
        '''I belive this is required for an Open AI gym env.'''
        pass


    def _find_space_limits(self):
        ''' find upper and lower bounds of action and observation spaces''' 

        torque_lb, torque_ub = self.quadruped.get_joint_torque_bounds()
        position_lb, position_ub = self.quadruped.get_joint_position_bounds()
        velocity_lb, velocity_ub = self.quadruped.get_joint_velocity_bounds()
        observation_lb = np.concatenate((torque_lb, 
                                        position_lb,
                                        velocity_lb, 
                                        -0.78 * np.ones(4), # this is for base orientation in quaternions
                                        np.zeros(4), # foot normal forces
                                        -1e5 * np.ones(3), # cartesian acceleration (arbitrary bound)
                                        -1e5 * np.ones(3), # angular velocity (arbitrary bound)
                                        -np.ones(self.grid_len**2) * (self.terrain_height + 5))) # 5 is a safe arbitrary value 

        observation_ub = np.concatenate((torque_ub, 
                                        position_ub, 
                                        velocity_ub, 
                                        0.78 * np.ones(4),
                                        1e4 * np.ones(4), # arbitrary bound
                                        1e5 * np.ones(3),
                                        1e5 * np.ones(3),
                                        np.ones(self.grid_len**2) * (self.terrain_height + 5)))


        return observation_lb, observation_ub
            

    def _reward_function(self) -> float:
        ''' Calculates reward based off of current state '''

        base_x_velocity = self.base_twist[0]
        torque_penalty = np.power(self.applied_torques, 2).mean()
        return base_x_velocity - 0.000005 * torque_penalty


    def _is_state_terminal(self) -> bool:
        ''' Calculates whether to end current episode due to failure based on current state. '''

        info = {}

        base_z_position = self.base_position[2]
        height_out_of_bounds = ((base_z_position < 0.2) or \
                                (base_z_position > 0.8 + self.terrain_height + self.terrain_height_range/2.))
        timeout = (self.eps_step_counter >= self.eps_timeout) or \
                    (self.base_position[0] >= self.terrain_length - 1.0)
        # I don't care about how much the robot yaws for termination, only if its flipped on its back.
        flipping = ((abs(np.array(p.getEulerFromQuaternion(self.base_orientation))) > [0.78*2, 0.78*2.5, 1e10]).any())
        y_out_of_bounds = not (-self.terrain_width/2. < self.base_position[1] < self.terrain_width/2.)

        if flipping:
            info['termination_reason'] = 'flipping'
        elif height_out_of_bounds:
            info['termination_reason'] = 'height_out_of_bounds'
        elif y_out_of_bounds:
            info['termination_reason'] = 'y_out_of_bounds'
        elif timeout: # {'TimeLimit.truncated': True}
            info['TimeLimit.truncated'] = True

        return any([flipping, height_out_of_bounds, timeout, y_out_of_bounds]), info


    def _update_state(self):

        self.joint_positions, self.joint_velocities, _, self.applied_torques = self.quadruped.get_joint_states()
        self.base_position, self.base_orientation = self.quadruped.get_base_position_and_orientation()
        self.base_twist = self.quadruped.get_base_twist()
        self.cartesian_base_accel = self.base_twist[:3] - self.previous_base_twist[:3] # TODO divide by timestep or assert timestep == 1/240.

        self.t = self.eps_step_counter * self.n_hold_frames / 240.

        self.foot_normal_forces = self._get_foot_contacts()
        
        self.state = np.concatenate((self.applied_torques, 
                                    self.joint_positions,
                                    self.joint_velocities,
                                    self.base_orientation,
                                    self.foot_normal_forces,
                                    self.cartesian_base_accel,
                                    self.base_twist[3:], # base angular velocity
                                    self._get_heightmap().flatten())) 
        
        if np.isnan(self.state).any():
            print('nans in state')
            breakpoint()

        # Not used in state, but used in _is_terminal() and _reward()    
        self.previous_base_twist = self.base_twist
    

if __name__ == '__main__':
    '''This test open the simulation in GUI mode for viewing the generated terrain, then saves a rendered image of each
    client for visual verification that the two are identical. Then the script just keeps generating random terrains 
    for viewing. '''

    env = gym.make('gym_aliengo:AliengoSteps-v0', render=True, realTime=True)
    imwrite('client_render.png', cvtColor(env.render(client=env.client, mode='rgb_array'), COLOR_RGB2BGR))
    imwrite('fake_client_render.png', cvtColor(env.render(client=env.fake_client, mode='rgb_array'), COLOR_RGB2BGR))

    
    while True:
        env.reset()
        time.sleep(5.0)


