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
from pybullet_utils import bullet_client as bc


'''A basic, flat-ground implementation of Aliengo for learning to walk. Only rewards forward motion with a small penalty
for joint torques. '''


class AliengoEnv(gym.Env):
    def __init__(self, render=False, realTime=False):
        # Environment Options
        self._apply_perturbations = False
        self.perturbation_rate = 0.00 # probability that a random perturbation is applied to the torso
        self.max_torque = 40.0 
        self.kp = 1.0 
        self.kd = 1.0
        self.n_hold_frames = 1 #TODO
        self._is_render = render
        self.eps_timeout = 240.0/self.n_hold_frames * 20 # number of steps to timeout after


        if self._is_render:
            self.client = bc.BulletClient(connection_mode=p.GUI)
        else:
            self.client = bc.BulletClient(connection_mode=p.DIRECT)

        if self.client == -1:
            raise RuntimeError('Pybullet could not connect to physics client')

        self.plane = self.client.loadURDF(os.path.join(os.path.dirname(__file__), '../urdf/plane.urdf'))
        self.quadruped = aliengo.Aliengo(pybullet_client=self.client, 
                                        max_torque=self.max_torque, 
                                        kp=self.kp, 
                                        kd=self.kd)

        self.client.setGravity(0,0,-9.8)
        self.client.setRealTimeSimulation(realTime) # this has no effect in DIRECT mode, only GUI mode
        self.client.setTimeStep(1/240.)

        # (50) applied torque, pos, and vel for each motor, base orientation (quaternions), foot normal forces,
        # cartesian base acceleration, base angular velocity
        self.state_space_dim = 12 * 3 + 4 + 4 + 3 + 3 
        self.num_joints = 18 # This includes fixed joints from the URDF
        self.action_space_dim = 12 # This is the number of Aliengo motors (3 per leg)


        self.state = np.zeros(self.state_space_dim) 
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


    def _is_non_foot_ground_contact(self):
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
        '''Returns a numpy array of shape (4,) containing the normal forces on each foot with the ground. '''

        contacts = [0] * 4
        for i in range(len(self.quadruped.foot_links)):
            info = self.client.getContactPoints(self.quadruped.quadruped, 
                                                self.plane, 
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
        # action = np.clip(action, self.action_space.low, self.action_space.high)
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
        done, info = self._is_state_terminal() # this must come after self._update_state()
        self.reward = self._reward_function() # this must come after self._update_state()

        # info = {'':''} # this is returned so that env.step() matches Open AI gym API
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


    def reset(self): #TODO add stochasticity to the initial starting state
        '''Resets the robot to a neutral standing position, knees slightly bent. The motor control command is to 
        prevent the robot from jumping/falling on first user command. Simulation is stepped to allow robot to fall
        to ground and settle completely.'''
        self.client.resetBasePositionAndOrientation(self.quadruped.quadruped,
                                            posObj=[0,0,0.48], 
                                            ornObj=[0,0,0,1.0]) 

        self.quadruped.reset_joint_positions(stochastic=True) # will put all joints at default starting positions
        for i in range(500): # to let the robot settle on the ground.
            self.client.stepSimulation()
        self._update_state()
        return self.state


    def render(self, mode='human'):
        '''Setting the render kwarg in the constructor determines if the env will render or not.'''

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
            view_matrix = self.client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_pos,
                distance=2.0,
                yaw=0,
                pitch=-30.,
                roll=0,
                upAxisIndex=2)
            proj_matrix = self.client.computeProjectionMatrixFOV(fov=60,
                aspect=float(RENDER_WIDTH) /
                RENDER_HEIGHT,
                nearVal=0.1,
                farVal=100.0)
            _, _, px, _, _ = self.client.getCameraImage(width=RENDER_WIDTH,
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
        return base_x_velocity - 0.001 * torque_penalty 



    def _is_state_terminal(self) -> bool:
        ''' Calculates whether to end current episode due to failure based on current state.
        Returns boolean and puts reason in info if True '''
        info = {}

        timeout = (self.eps_step_counter >= self.eps_timeout)
        base_z_position = self.base_position[2]
        height_out_of_bounds = ((base_z_position < 0.23) or (base_z_position > 0.8)) 
        falling = ((abs(np.array(p.getEulerFromQuaternion(self.base_orientation))) > [0.78*2, 0.78, 0.78]).any()) 

        if falling:
            info['termination_reason'] = 'falling'
        elif height_out_of_bounds:
            info['termination_reason'] = 'height_out_of_bounds'
        elif timeout: # {'TimeLimit.truncated': True}
            info['TimeLimit.truncated'] = True

        return any([falling, height_out_of_bounds, timeout]), info


    def _update_state(self):

        self.joint_positions, self.joint_velocities, _, self.applied_torques = self.quadruped.get_joint_states()
        self.base_position, self.base_orientation = self.quadruped.get_base_position_and_orientation()
        self.base_twist = self.quadruped.get_base_twist()
        self.cartesian_base_accel = self.base_twist[:3] - self.previous_base_twist[:3] # TODO divide by timestep or assert timestep == 1/240.
        self.foot_normal_forces = self._get_foot_contacts()
        self.state = np.concatenate((self.applied_torques, 
                                    self.joint_positions,
                                    self.joint_velocities,
                                    self.base_orientation,
                                    self.foot_normal_forces,
                                    self.cartesian_base_accel,
                                    self.base_twist[3:])) # last item is base angular velocity
        

        if np.isnan(self.state).any():
            print('nans in state')
            breakpoint()

        # Not used in state, but used in _is_terminal() and _reward()    
        self.previous_base_twist = self.base_twist
    

if __name__ == '__main__':
    '''Perform check by feeding in the mocap trajectory provided by Unitree (linked) into the aliengo robot and
    save video. https://github.com/unitreerobotics/aliengo_pybullet'''

    import cv2
    env = gym.make('gym_aliengo:Aliengo-v0')    
    env.reset()

    img = env.render('rgb_array')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_list = [img]
    counter = 0

    with open('mocap.txt','r') as f:
        for line_num, line in enumerate(f): 
            if line_num%2 == 0: # Unitree runs this demo at 500 Hz. We run at 240 Hz, so double is close enough.
                action = env.quadruped._positions_to_actions(np.array(line.split(',')[2:],dtype=np.float32))
                obs,_ , done, _ = env.step(action)
                if counter%4 == 0:  # simulation runs at 240 Hz, so if we render every 4th frame, we get 60 fps video
                    img = env.render('rgb_array')
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img_list.append(img)
                counter +=1 # only count the lines that are sent to the simulation (i.e. only count 
                # p.client.stepSimulation() calls)

    height, width, layers = img.shape
    size = (width, height)
    out = cv2.VideoWriter('test_vid.avi', cv2.VideoWriter_fourcc(*'XVID'), 60, size)
    for img in img_list:
        out.write(img)
    out.release()
    print('Video saved')


    
    



    


