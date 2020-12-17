import gym
from gym import error, spaces, utils
from gym.utils import seeding
import sys
import pybullet as p
import os
import time
import numpy as np
import warnings
from cv2 import putText, FONT_HERSHEY_SIMPLEX
from gym_aliengo.envs import aliengo


class AliengoSteppingStones(gym.Env):
    def __init__(self, render=False, realTime=False):
        # Environment Options
        self._apply_perturbations = False
        self.perturbation_rate = 0.00 # probability that a random perturbation is applied to the torso
        self.max_torque = 40.0 
        self.kp = 1.0 
        self.kd = 1.0
        self.n_hold_frames = 1 
        self._is_render = render
        self.eps_timeout = 240.0/self.n_hold_frames * 20 # number of steps to timeout after

        # stepping stone parameters
        self.height = 1.0 # height of the heightfield
        self.course_length = 10.0 # total distance from edge of start block to edge of end block
        self.course_width = 2.0 # widght of path of stepping stones
        self.stone_length = 0.25 # side length of square stepping stones
        self.stone_density = 6.0 # stones per square meter
        self.stone_height_range = 0.25 # heights of stones will be within [self.height - this/2, self.height + this/2 ]

        # heightmap parameters
        self.length = 1.25# assumes square
        self.robot_position = 0.5 # distance of robot base origin from back edge of height map
        self.grid_spacing = 0.125 
        assert self.length%self.grid_spacing == 0


        if self._is_render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        self.fake_client = p.connect(p.DIRECT) # this is only used for getting the heightmap

        if self.client == -1:
            raise RuntimeError('Pybullet could not connect to physics client')

        # urdfFlags = p.URDF_USE_SELF_COLLISION
        self.plane = p.loadURDF(os.path.join(os.path.dirname(__file__), '../urdf/plane.urdf'), 
                                physicsClientId=self.client)

        self.quadruped = aliengo.Aliengo(pybullet_client=self.client, 
                                        max_torque=self.max_torque, 
                                        kp=self.kp, 
                                        kd=self.kd)


        p.setGravity(0,0,-9.8, physicsClientId=self.client)
        # self.quadruped.foot_links = [5, 9, 13, 17]
        # self.lower_legs = [2,5,8,11]
        # for l0 in self.lower_legs:
        #     for l1 in self.lower_legs:
        #         if (l1>l0):
        #             enableCollision = 1
        #             # print("collision for pair",l0,l1, p.getJointInfo(self.quadruped,l0, physicsClientId=self.client)[12],p.getJointInfo(self.quadruped,l1, physicsClientId=self.client)[12], "enabled=",enableCollision)
        #             p.setCollisionFilterPair(self.quadruped, self.quadruped, l0,l1,enableCollision, physicsClientId=self.client)

        p.setRealTimeSimulation(realTime, physicsClientId=self.client) # this has no effect in DIRECT mode, only GUI mode
        # below is the default. Simulation params need to be retuned if this is changed
        p.setTimeStep(1/240., physicsClientId=self.client)

        # for i in range (p.getNumJoints(self.quadruped, physicsClientId=self.client)):
        #     p.changeDynamics(self.quadruped,i,linearDamping=0, angularDamping=.5, physicsClientId=self.client)



        # (50) applied torque, pos, and vel for each motor, base orientation (quaternions), foot normal forces,
        # cartesian base acceleration, base angular velocity
        self.state_space_dim = 12 * 3 + 4 + 4 + 3 + 3 + int(self.length/self.grid_spacing)  
        self.num_joints = 18 # This includes fixed joints from the URDF
        self.action_space_dim = self.quadruped.n_motors # this remains unchanged

        self.state = np.zeros(self.state_space_dim) # I currently don't distinguish between state and observation
        self.applied_torques = np.zeros(self.quadruped.n_motors) 
        self.joint_velocities = np.zeros(self.quadruped.n_motors)
        self.joint_positions = np.zeros(self.quadruped.n_motors)
        self.base_orientation = np.zeros(4)
        self.foot_normal_forces = np.zeros(4)
        self.cartesian_base_accel = np.zeros(3) 
        self.base_twist = np.zeros(6) # used to calculate accelerations, angular vel included in state
        self.previous_base_twist = np.zeros(6) # used to calculate accelerations, angular vel included in state
        self.base_position = np.zeros(3) # not returned as observation, but used for calculating reward or termination
        self.eps_step_counter = 0 # Used for triggering timeout
        self.t = 0 # represents the actual time


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

        self._first_run = True # used so that I don't call _remove_stepping_stones on the first run
        self._stone_ids = []


    def _is_non_foot_ground_contact(self): #TODO if I ever use this in this env, account for stepping stone contact
        """Detect if any parts of the robot, other than the feet, are touching the ground."""

        contact = False
        for i in range(self.num_joints):
            if i in self.quadruped.foot_links: # the feet themselves are allow the touch the ground
                continue
            points = p.getContactPoints(self.quadruped.quadruped, self.plane, linkIndexA=i, physicsClientId=self.client)
            if len(points) != 0:
                contact = True
        return contact


    def _get_foot_contacts(self): 
        '''Returns a numpy array of shape (4,) containing the normal forces on each foot with the ground. '''

        contacts = [0] * 4
        for i in range(len(self.quadruped.foot_links)):
            info = p.getContactPoints(bodyA=self.quadruped.quadruped,  
                                        linkIndexA=self.quadruped.foot_links[i],
                                        physicsClientId=self.client)
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
            p.stepSimulation(physicsClientId=self.client)
        self.eps_step_counter += 1
        self._update_state()
        done, info = self._is_state_terminal()
        self.reward = self._reward_function()

        # info = {'':''} # this is returned so that env.step() matches Open AI gym API
        if done:
            self.eps_step_counter = 0
            # if self.trot_prior:
                # info['avg_trot_loss'] = self.trot_loss_history.mean()
                # self.trot_loss_history = np.array([])
        return self.state, self.reward, done, info

        
    def _apply_perturbation(self):
        raise NotImplementedError
        if np.random.rand() > 0.5: # apply force
            force = tuple(10 * (np.random.rand(3) - 0.5))
            p.applyExternalForce(self.quadruped, -1, force, (0,0,0), p.LINK_FRAME, physicsClientId=self.client)
        else: # apply torque
            torque = tuple(0.5 * (np.random.rand(3) - 0.5))
            p.applyExternalTorque(self.quadruped, -1, torque, p.LINK_FRAME, physicsClientId=self.client)


    def reset(self): #TODO add stochasticity
        '''Resets the robot to a neutral standing position, knees slightly bent. The motor control command is to 
        prevent the robot from jumping/falling on first user command. Simulation is stepped to allow robot to fall
        to ground and settle completely.'''

        if not self._first_run:
            self._remove_stepping_stones()
        else:
            self._first_run = False
        self._create_stepping_stones()

        p.resetBasePositionAndOrientation(self.quadruped.quadruped,
                                            posObj=[0,0,self.height + 0.48], 
                                            ornObj=[0,0,0,1.0],
                                            physicsClientId=self.client) 

        self.quadruped.reset_joint_positions() # will put all joints at default starting positions
        for i in range(500): # to let the robot settle on the ground.
            p.stepSimulation(physicsClientId=self.client)
        self._update_state()
        return self.state
    
    
    def _remove_stepping_stones(self):
        for id in self._stone_ids:
            p.removeBody(id)
        self._stone_ids = []

    
    def _create_stepping_stones(self):
        start_block = p.createCollisionShape(p.GEOM_BOX, 
                                            halfExtents=[1, self.course_width/2.0, self.height/2.0], 
                                            physicsClientId=self.client)
        stepping_stone = p.createCollisionShape(p.GEOM_BOX, 
                                            halfExtents=[self.stone_length/2.0, self.stone_length/2.0, self.height/2.0], 
                                            physicsClientId=self.client)
        start_body = p.createMultiBody(baseCollisionShapeIndex=start_block, basePosition=[0,0,self.height/2.0])
        end_body = p.createMultiBody(baseCollisionShapeIndex=start_block, basePosition=[self.course_length + 2.0,
                                                                                    0,self.height/2.])
        
        n_stones = int(self.course_length * self.course_width * self.stone_density)
        stone_heights = (np.random.rand(n_stones) - 0.5) * self.stone_height_range + self.height/2.0 
        stone_x = np.random.rand(n_stones) * self.course_length + 1.0
        stone_y = (np.random.rand(n_stones) - 0.5) * self.course_width

        self._stone_ids = [start_body, end_body]
        for i in range(n_stones):
            id = p.createMultiBody(baseCollisionShapeIndex=stepping_stone, 
                                    basePosition=[stone_x[i], stone_y[i], stone_heights[i]])
            self._stone_ids.append(id)

    
    def _get_heightmap(self):
        '''Debug flag enables printing of labeled coordinates and measured heights to rendered simulation.'''

        debug = False

        base_x = self.base_position[0]
        base_y = self.base_position[1]
        base_z = self.base_position[2]

        grid_len = int(self.length/self.grid_spacing) + 1
        x = np.linspace(0, self.length, grid_len)
        y = np.linspace(-self.length/2.0, self.length/2.0, grid_len)
        coordinates = np.array(np.meshgrid(x,y))
        coordinates[0,:,:] += base_x - self.robot_position
        coordinates[1,:,:] += base_y  
        # coordinates has shape (2, grid_len, grid_len)
        coor_list = coordinates.reshape((2, grid_len**2)).swapaxes(0, 1) # is now shape (grid_len*grid_len,2) 
        ray_start = np.append(coor_list, np.ones((grid_len**2, 1)) * (self.height + self.stone_height_range), axis=1)
        ray_end = np.append(coor_list, np.zeros((grid_len**2, 1)) - 1, axis=1)
        raw_output = p.rayTestBatch(ray_start, ray_end, physicsClientId=self.client) #TODO I need the height map to ignore the actual robot
        # it ignores the torso, but I also want it to ignore the legs and all. Just create another simulation instance
        z_heights = np.array([raw_output[i][3][2] for i in range(grid_len**2)])
        relative_z_heights = z_heights - base_z
        if debug:
            p.addUserDebugText(text='%.2f, %.2f'%(base_x, base_y),
                        textPosition=[base_x, base_y,self.height+1],
                        textColorRGB=[0,0,0])
            for i in range(grid_len):
                for j in range(grid_len):
                    p.addUserDebugText(
                        text='%.2f, %.2f, %.2f'%(coordinates[0,i,j], coordinates[1,i,j], z_heights.reshape((grid_len, grid_len))[i,j]),
                        textPosition=[coordinates[0,i,j], coordinates[1,i,j],self.height+1],
                        textColorRGB=[0,0,0]
                        )
                    p.addUserDebugLine( [coordinates[0,i,j], coordinates[1,i,j],self.height+1],
                                        [coordinates[0,i,j], coordinates[1,i,j], 0],
                                        lineColorRGB=[0,0,0] )
        
        return relative_z_heights.reshape((grid_len, grid_len))
        

    def render(self, mode='human'):
        '''Setting the render kwarg in the constructor determines if the env will render or not.'''

        RENDER_WIDTH = 480 
        RENDER_HEIGHT = 360

        base_x_velocity = np.array(p.getBaseVelocity(self.quadruped.quadruped, 
                                    physicsClientId=self.client)).flatten()[0]
        torque_pen = -0.00001 * np.power(self.applied_torques, 2).mean()

        # RENDER_WIDTH = 960 
        # RENDER_HEIGHT = 720

        # RENDER_WIDTH = 1920
        # RENDER_HEIGHT = 1080

        if mode == 'rgb_array':
            base_pos, _ = p.getBasePositionAndOrientation(self.quadruped.quadruped, physicsClientId=self.client)
            # base_pos = self.minitaur.GetBasePosition()
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_pos,
                distance=2.0,
                yaw=0,
                pitch=-30.,
                roll=0,
                upAxisIndex=2,
                physicsClientId=self.client)
            proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                aspect=float(RENDER_WIDTH) /
                RENDER_HEIGHT,
                nearVal=0.1,
                farVal=100.0,
                physicsClientId=self.client)
            _, _, px, _, _ = p.getCameraImage(width=RENDER_WIDTH,
                height=RENDER_HEIGHT,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                physicsClientId=self.client)
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
            img = putText(np.float32(img), 
                            'Body Contact: ' + str(self._is_non_foot_ground_contact()), 
                            (200, 60 + 20 * 4), 
                            FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
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
                                        -1e5 * np.ones(3))) # angular velocity (arbitrary bound)

        observation_ub = np.concatenate((torque_ub, 
                                        position_ub, 
                                        velocity_ub, 
                                        0.78 * np.ones(4),
                                        1e4 * np.ones(4), # arbitrary bound
                                        1e5 * np.ones(3),
                                        1e5 * np.ones(3)))


        return observation_lb, observation_ub
            

    def _reward_function(self) -> float:
        ''' Calculates reward based off of current state '''

        base_x_velocity = self.base_twist[0]
        torque_penalty = np.power(self.applied_torques, 2).mean()
        finish_bonus = (self.base_position[0] >= self.course_length + 2) * 20 #TODO check this
        return base_x_velocity - 0.000005 * torque_penalty + finish_bonus



    def _is_state_terminal(self) -> bool:
        ''' Calculates whether to end current episode due to failure based on current state.
        Returns boolean and puts reason in info if True '''
        info = {}

        timeout = (self.eps_step_counter >= self.eps_timeout)


        base_z_position = self.base_position[2]
        height_out_of_bounds = ((base_z_position < 0.23) or (base_z_position > 0.8)) 
        body_contact = self._is_non_foot_ground_contact() * 0
        # 0.78 rad is about 45 deg
        falling = ((abs(np.array(p.getEulerFromQuaternion(self.base_orientation))) > [0.78*2, 0.78, 0.78]).any()) 
        # falling = (abs(np.array(p.getEulerFromQuaternion(self.base_orientation))) > 0.78).any() 
        going_backwards = (self.base_twist[0] <= -1.0) * 0
        self_collision = self.quadruped.self_collision() * 0

        no_feet_on_ground = (self.foot_normal_forces == 0).all() * 0
        if falling:
            info['termination_reason'] = 'falling'
        elif height_out_of_bounds:
            info['termination_reason'] = 'height_out_of_bounds'
        elif body_contact:
            info['termination_reason'] = 'body_contact_with_ground'
        elif going_backwards:
            info['termination_reason'] = 'going_backwards'
        elif self_collision:
            info['termination_reason'] = 'self_collision'
        elif no_feet_on_ground:
            info['termination_reason'] = 'no_feet_on_ground'
        elif timeout: # {'TimeLimit.truncated': True}
            info['TimeLimit.truncated'] = True

        return any([falling, height_out_of_bounds, body_contact, going_backwards, self_collision, no_feet_on_ground, \
            timeout]), info


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
                                    self.base_twist[3:],
                                    self._get_heightmap().flatten())) # last item is base angular velocity
        
        if np.isnan(self.state).any():
            print('nans in state')
            breakpoint()

        # Not used in state, but used in _is_terminal() and _reward()    
        self.previous_base_twist = self.base_twist
    

if __name__ == '__main__':
    env = gym.make('gym_aliengo:AliengoSteppingStones-v0', render=True, realTime=True)
    env.reset()
    time.sleep(5)
    env.reset()
    # env._get_heightmap()
    while True:
        time.sleep(1)


