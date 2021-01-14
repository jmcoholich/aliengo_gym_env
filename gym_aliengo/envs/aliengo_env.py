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
    def __init__(self, render=False, realTime=False, use_pmtg=True):
        # Environment Options
        self._apply_perturbations = False
        self.perturbation_rate = 0.00 # probability that a random perturbation is applied to the torso
        self.max_torque = 40.0 
        self.kp = 0.1
        self.kd = 1.0
        self.n_hold_frames = 4
        self._is_render = render
        self.eps_timeout = 240.0/self.n_hold_frames * 20 # number of steps to timeout after
        self.use_pmtg = use_pmtg
        self.speed_clipping = 2.0 # "...maximum walking speed exceeds 1.5 m/s" https://www.unitree.com/products/aliengo


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

        if self.use_pmtg:
            # state space consists of sin(phase) and cos(phase) for each leg, 4D IMU data, last position targets
            # Note that the (more advanced) Hutter implementation uses a larger state. For flat ground this is fine. 
            # Also see the original PMTG paper: https://arxiv.org/pdf/1910.02812.pdf
            self.state_space_dim = 8 + 4 + 12 # 18 
            self.action_space_dim = 16 # 4 frequency adjustments per leg, 12 position residuals (xyz per leg)
            self.action_lb, self.action_ub = self.quadruped.get_pmtg_bounds() 
            self.t = 0.0
            self.last_foot_position_command = np.zeros((4,3)) # This will actually be initialized when reset() is called
            self.phases = np.zeros(4)
        else:
            # (50) applied torque, pos, and vel for each motor, base orientation (quaternions), foot normal forces,
            # cartesian base acceleration, base angular velocity
            self.state_space_dim = 12 * 3 + 4 + 4 + 3 + 3 
            self.action_space_dim = 12 # This is the number of Aliengo motors (3 per leg)
            self.action_lb, self.action_ub = self.quadruped.get_joint_position_bounds()
        self.num_joints = 18 # This includes fixed joints from the URDF

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
        self.mean_torque_squared = 0 # online avg
        self.mean_x_vel = 0 # online avg
        # TODO add this^ to other envs
            
        self.action_space = spaces.Box(
            low=self.action_lb,
            high=self.action_ub,
            dtype=np.float32
            )
        observation_lb, observation_ub = self._find_space_limits() # this function includes logic for self.use_pmtg
        self.observation_space = spaces.Box(
            low=observation_lb,
            high=observation_ub,
            dtype=np.float32
            )


    def step(self, action):
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        DELTA = 0.01
        if not ((self.action_lb - DELTA <= action) & (action <= self.action_ub + DELTA)).all():
            print("Action passed to env.step(): ", action)
            raise ValueError('Action is out-of-bounds of:\n' + str(self.action_lb) + '\nto\n' + str(self.action_ub)) 

        if self.use_pmtg:
            f = action[:4]
            residuals = action[4:].reshape((4,3))
            self.phases, self.last_foot_position_command = self.quadruped.set_trajectory_parameters(self.t, 
                                                                                                    f=f, 
                                                                                                    residuals=residuals)
            self.t += 1./240. * self.n_hold_frames
        else:
            self.quadruped.set_joint_position_targets(action)

        if (np.random.rand() > self.perturbation_rate) and self._apply_perturbations: 
            raise NotImplementedError
            self._apply_perturbation()
        for _ in range(self.n_hold_frames):
            self.client.stepSimulation()
        self.eps_step_counter += 1
        self._update_state()
        done, info = self._is_state_terminal() # this must come after self._update_state()
        self.reward, torque_penalty = self._reward_function() # this must come after self._update_state()
        self.mean_torque_squared += (torque_penalty - self.mean_torque_squared)/self.eps_step_counter 
        self.mean_x_vel += (self.base_twist[0] - self.mean_x_vel)/self.eps_step_counter

        # info = {'':''} # this is returned so that env.step() matches Open AI gym API 
        if done:
            info['mean_torque_squared'] = self.mean_torque_squared
            info['distance_traveled'] = self.base_position[0]
            info['mean_x_vel'] = self.mean_x_vel

        return self.state, self.reward, done, info

        
    def reset(self): 
        '''Resets the robot to a neutral standing position, knees slightly bent. The motor control command is to 
        prevent the robot from jumping/falling on first user command. Simulation is stepped to allow robot to fall
        to ground and settle completely.'''

        self.eps_step_counter = 0 # TODO do this in other envs
        self.client.resetBasePositionAndOrientation(self.quadruped.quadruped,
                                            posObj=[0,0,0.48], 
                                            ornObj=[0,0,0,1.0]) 

        foot_positions = self.quadruped.reset_joint_positions(stochastic=True) 
        for i in range(500): # to let the robot settle on the ground.
            self.client.stepSimulation()
        if self.use_pmtg:
            self.t = 0.0
            self.last_foot_position_command = foot_positions 
        self._update_state()
        return self.state


    def render(self, mode='human'):
        return self.quadruped.render(mode=mode, client=self.client)


    def close(self):
        '''I belive this is required for an Open AI gym env.'''

        pass


    def _find_space_limits(self):
        ''' find upper and lower bounds of action and observation spaces''' 

        if self.use_pmtg:
            # state space consists of sin(phase) and cos(phase) for each leg, 4D IMU data, last position targets
            # Note that the (more advanced) Hutter implementation uses a larger state. For flat ground simple is fine.
            # 4D IMU data is pitch, roll, pitch rate, roll rate.
            observation_ub = np.concatenate((np.ones(8), 
                                                np.array([np.pi, np.pi, 1e5, 1e5]), # pitch, roll, pitch rate, roll rate
                                                np.ones(12)*2)) # last foot position commands
            return -observation_ub, observation_ub
        else:
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
        ''' Calculates reward based off of current state. Uses reward clipping on speed. '''

        base_x_velocity = np.clip(self.base_twist[0], -np.inf, self.speed_clipping)
        torque_penalty = (self.applied_torques * self.applied_torques).mean() # TODO do in other envs
        return base_x_velocity - 0.0001 * torque_penalty, torque_penalty #TODO return value of torque penalty and distance traveled etc
        # TODO return torque penality without the coefficient, so I can compare values objectively, even if I later change
        # the coefficient # change the baseline Kostrikov to just log any extra shid in the returned info dict


    def _is_state_terminal(self) -> bool:
        ''' Calculates whether to end current episode due to failure based on current state.
        Returns boolean and puts reason in info if True '''
        info = {}

        timeout = (self.eps_step_counter >= self.eps_timeout)
        base_z_position = self.base_position[2]
        height_out_of_bounds = ((base_z_position < 0.23) or (base_z_position > 0.8)) 
        falling = ((abs(np.array(p.getEulerFromQuaternion(self.base_orientation))) > \
                                                                                [np.pi/2., np.pi/4., np.pi/4.]).any()) 

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
        self.cartesian_base_accel = (self.base_twist[:3] - self.previous_base_twist[:3])/(1.0/240 * self.n_hold_frames)
        self.foot_normal_forces = self.quadruped._get_foot_contacts()

        if self.use_pmtg:
            # state space consists of sin(phase) and cos(phase) for each leg, 4D IMU data, last position targets
            self.state = np.concatenate((np.sin(self.phases),
                                            np.cos(self.phases),
                                            self.client.getEulerFromQuaternion(self.base_orientation)[:-1], # roll,pitch
                                            self.base_twist[3:-1],# roll rate, pitch rate
                                            self.last_foot_position_command.flatten()))
        else:
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
    env = gym.make('gym_aliengo:Aliengo-v0', use_pmtg=False)    
    env.reset()

    img = env.render('rgb_array')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_list = [img]
    counter = 0

    with open('mocap.txt','r') as f:
        for line_num, line in enumerate(f): 
            if line_num%2*env.n_hold_frames == 0: # Unitree runs this demo at 500 Hz. We run at 240 Hz, so double is close enough.
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


    
    



    


