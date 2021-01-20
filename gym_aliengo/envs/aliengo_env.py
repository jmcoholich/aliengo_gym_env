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
    def __init__(self, 
                    render=False, 
                    use_pmtg=True,
                    apply_perturb=True,
                    avg_time_per_perturb=5.0, # seconds
                    max_torque=40.0, # N-m
                    kp=0.1, # acts on position erorr
                    kd=1.0, # acts on rate of position erorr
                    action_repeat=4,
                    timeout=20.0, # number of seconds to timeout after
                    realTime=False): # should never be True when training, only for visualzation or debugging MAYBE
        # Environment Options
        self.use_pmtg = use_pmtg
        self.apply_perturb = apply_perturb
        self.avg_time_per_perturb = avg_time_per_perturb # average time in seconds between perturbations 
        self.n_hold_frames = action_repeat
        self.eps_timeout = 240.0/self.n_hold_frames * timeout # number of steps to timeout after
        # if U[0, 1) is less than this, perturb
        self.perturb_p = 1.0/(self.avg_time_per_perturb * 240.0) * self.n_hold_frames


        if render:
            self.client = bc.BulletClient(connection_mode=p.GUI)
        else:
            self.client = bc.BulletClient(connection_mode=p.DIRECT)

        if self.client == -1: # not 100% sure that BulletClient class works like this, but will leave this here for now
            raise RuntimeError('Pybullet could not connect to physics client')

        self.plane = self.client.loadURDF(os.path.join(os.path.dirname(__file__), '../urdf/plane.urdf'))
        self.quadruped = aliengo.Aliengo(pybullet_client=self.client, 
                                            max_torque=max_torque, 
                                            kp=kp, 
                                            kd=kd)
        self.client.setGravity(0,0,-9.8)
        if realTime:
            warnings.warn('\n\n' + '#'*100 + '\nExternal force/torque disturbances will NOT work properly with '
                            'real time pybullet GUI enabled.\n' + '#'*100 + '\n') # how to make this warning yellow?
        self.client.setRealTimeSimulation(realTime) # this has no effect in DIRECT mode, only GUI mode
        self.client.setTimeStep(1/240.0)

        if self.use_pmtg:
            # state space consists of sin(phase) and cos(phase) for each leg, 4D IMU data, last position targets
            # Note that the (more advanced) Hutter implementation uses a larger state. For flat ground this is fine. 
            # Also see the original PMTG paper: https://arxiv.org/pdf/1910.02812.pdf
            # self.state_space_dim = 8 + 4 + 12 # 18 
            # self.action_space_dim = 16 # 4 frequency adjustments per leg, 12 position residuals (xyz per leg)
            self.action_lb, self.action_ub = self.quadruped.get_pmtg_action_bounds() 
            observation_lb, observation_ub = self.quadruped.get_pmtg_observation_bounds()
            self.t = 0.0
            # self.last_foot_position_command = np.zeros((4,3)) # This will actually be initialized when reset() is called
            # self.phases = np.zeros(4)
        else:
            # (50) applied torque, pos, and vel for each motor, base orientation (quaternions), foot normal forces,
            # cartesian base acceleration, base angular velocity
            # self.state_space_dim = 12 * 3 + 4 + 4 + 3 + 3 
            # self.action_space_dim = 12 # This is the number of Aliengo motors (3 per leg)
            self.action_lb, self.action_ub = self.quadruped.get_joint_position_bounds()
            observation_lb, observation_ub = self.quadruped.get_observation_bounds()

        # self.state = np.zeros(self.state_space_dim) 
        # self.applied_torques = np.zeros(12)
        # self.joint_velocities = np.zeros(12)
        # self.joint_positions = np.zeros(12)
        # self.base_orientation = np.zeros(4)
        # self.foot_normal_forces = np.zeros(4)
        # self.cartesian_base_accel = np.zeros(3) 
        # self.base_twist = np.zeros(6) # used to calculate accelerations, angular vel included in state
        # self.previous_base_twist = np.zeros(6) # used to calculate accelerations, angular vel included in state
        # self.base_position = np.zeros(3) # not returned as observation, but used for calculating reward or termination
        self.eps_step_counter = 0 # Used for triggering timeout
        self.mean_rew_dict = {} # used for logging the mean reward terms at the end of each episode
        self.action_space = spaces.Box(
            low=self.action_lb,
            high=self.action_ub,
            dtype=np.float32
            )
        self.observation_space = spaces.Box(
            low=observation_lb,
            high=observation_ub,
            dtype=np.float32
            )
        # observation_lb, observation_ub = self._find_space_limits() # this function includes logic for self.use_pmtg


    def step(self, action):
        DELTA = 0.01
        if not ((self.action_lb - DELTA <= action) & (action <= self.action_ub + DELTA)).all():
            print("Action passed to env.step(): ", action)
            raise ValueError('Action is out-of-bounds of:\n' + str(self.action_lb) + '\nto\n' + str(self.action_ub)) 

        if self.use_pmtg:
            f = action[:4]
            residuals = action[4:].reshape((4,3))
            self.quadruped.set_trajectory_parameters(self.t, f=f, residuals=residuals)
            self.t += 1./240. * self.n_hold_frames
        else:
            self.quadruped.set_joint_position_targets(action)

        if (np.random.rand() < self.perturb_p) and self.apply_perturb: 
            '''TODO eventually make disturbance generating function that applies disturbances for multiple timesteps'''
            if np.random.rand() > 0.5:
                # TODO returned values will be part of privledged information for teacher training
                force, foot = self.quadruped.apply_foot_disturbance() 
            else:
                # TODO returned values will be part of privledged information for teacher training
                wrench = self.quadruped.apply_torso_disturbance()

        for _ in range(self.n_hold_frames): self.client.stepSimulation()
        self.eps_step_counter += 1
        self.quadruped.update_state()

        if self.use_pmtg:
            obs = self.quadruped.get_pmtg_observation()
        else:
            obs = self.quadruped.get_observation()

        info = {}
        done, termination_dict = self._is_state_terminal() # this must come after self._update_state()
        info.update(termination_dict) # termination_dict is an empty dict if not done

        if self.use_pmtg:
            rew, rew_dict = self.quadruped.pmtg_reward()
        else:
            raise NotImplementedError
            rew, rew_dict = self.quadruped.reward()
        self._update_mean_rew_dict(rew_dict)

        if done:
            info['distance_traveled']   = self.quadruped.base_position[0]
            info.update(self.mean_rew_dict)

        return obs, rew, done, info


    def _update_mean_rew_dict(self, rew_dict):
        '''Update self.mean_rew_dict, which keeps a running average of all terms of the reward. At the end of the 
        episode, the average will be logged. '''

        if self.eps_step_counter == 1:
            for key in rew_dict:  
                self.mean_rew_dict['mean' + key] = rew_dict[key]
        elif self.eps_step_counter > 1:
            for key in rew_dict:
                self.mean_rew_dict['mean' + key] += \
                                        (rew_dict[key] - self.mean_rew_dict['mean' + key])/float(self.eps_step_counter)
        else:
            assert False
            

    def reset(self): 
        '''Resets the robot to a neutral standing position, knees slightly bent. The motor control command is to 
        prevent the robot from jumping/falling on first user command. '''

        self.eps_step_counter = 0
        self.client.resetBasePositionAndOrientation(self.quadruped.quadruped,
                                            posObj=[0,0,0.48], 
                                            ornObj=[0,0,0,1.0]) 

        self.quadruped.reset_joint_positions(stochastic=True) 
        for i in range(500): # to let the robot settle on the ground.
            self.client.stepSimulation()
        self.quadruped.update_state()
        if self.use_pmtg:
            self.t = 0.0
            obs = self.quadruped.get_pmtg_observation()
        else:
            obs = self.quadruped.get_observation()
        return obs


    def render(self, mode='human'):
        return self.quadruped.render(mode=mode, client=self.client)


    def close(self):
        '''I belive this is required for an Open AI gym env.'''

        pass


    # def _find_space_limits(self):
    #     ''' find upper and lower bounds of action and observation spaces''' 

    #     if self.use_pmtg:
    #         # state space consists of sin(phase) and cos(phase) for each leg, 4D IMU data, last position targets
    #         # Note that the (more advanced) Hutter implementation uses a larger state. For flat ground simple is fine.
    #         # 4D IMU data is pitch, roll, pitch rate, roll rate.
    #         observation_ub = np.concatenate((np.ones(8), 
    #                                             np.array([np.pi, np.pi, 1e5, 1e5]), # pitch, roll, pitch rate, roll rate
    #                                             np.ones(12)*2)) # last foot position commands
    #         return -observation_ub, observation_ub
    #     else:
    #         torque_lb, torque_ub = self.quadruped.get_joint_torque_bounds()
    #         position_lb, position_ub = self.quadruped.get_joint_position_bounds()
    #         velocity_lb, velocity_ub = self.quadruped.get_joint_velocity_bounds()
    #         observation_lb = np.concatenate((torque_lb, 
    #                                         position_lb,
    #                                         velocity_lb, 
    #                                         -0.78 * np.ones(4), # this is for base orientation in quaternions
    #                                         np.zeros(4), # foot normal forces
    #                                         -1e5 * np.ones(3), # cartesian acceleration (arbitrary bound)
    #                                         -1e5 * np.ones(3))) # angular velocity (arbitrary bound)

    #         observation_ub = np.concatenate((torque_ub, 
    #                                         position_ub, 
    #                                         velocity_ub, 
    #                                         0.78 * np.ones(4),
    #                                         1e4 * np.ones(4), # arbitrary bound
    #                                         1e5 * np.ones(3),
    #                                         1e5 * np.ones(3)))

    #         return observation_lb, observation_ub
                

    # def _reward_function(self) -> float:
    #     ''' Calculates reward based off of current state. Uses reward clipping on speed. '''

    #     # base_x_velocity = np.clip(self.base_twist[0], -np.inf, self.speed_clipping)
    #     # torque_penalty = (self.applied_torques * self.applied_torques).mean() 
    #     if self.use_pmtg: 
    #         return self.quadruped.pmtg_reward()
    #     else:
    #         raise NotImplementedError
    #         rew, rew_terms_dict = self.quadruped.reward() #TODO
    #     return base_x_velocity - 0.0001 * torque_penalty, rew_terms_dict 


    def _is_state_terminal(self):
        quadruped_done, termination_dict = self.quadruped.is_state_terminal()
        timeout = (self.eps_step_counter >= self.eps_timeout) 
        if timeout:
            termination_dict['TimeLimit.truncated'] = True
        done = quadruped_done or timeout
        return done, termination_dict
        
        
    # def _update_state(self):

    #     self.joint_positions, self.joint_velocities, _, self.applied_torques = self.quadruped.get_joint_states()
    #     self.base_position, self.base_orientation = self.quadruped.get_base_position_and_orientation()
    #     self.base_twist = self.quadruped.get_base_twist()
    #     self.cartesian_base_accel = (self.base_twist[:3] - self.previous_base_twist[:3])/(1.0/240 * self.n_hold_frames)
    #     self.foot_normal_forces = self.quadruped.get_foot_contacts()

    #     if self.use_pmtg:
    #         # state space consists of sin(phase) and cos(phase) for each leg, 4D IMU data, last position targets
    #         imu = np.concatenate((self.client.getEulerFromQuaternion(self.base_orientation)[:-1], 
    #                                 self.base_twist[3:-1]))
    #         # std of pitch and roll noise is 0.9 deg, std of pitch rate and roll rate is 1.8 deg/s
    #         imu += np.random.randn(4) * np.array([np.pi/2. * 0.01]*2 + [np.pi * 0.01]*2) 
    #         self.state = np.concatenate((np.sin(self.phases),
    #                                         np.cos(self.phases),
    #                                         imu,
    #                                         self.last_foot_position_command.flatten()))
    #     else:
    #         self.state = np.concatenate((self.applied_torques, 
    #                                         self.joint_positions,
    #                                         self.joint_velocities,
    #                                         self.base_orientation,
    #                                         self.foot_normal_forces,
    #                                         self.cartesian_base_accel,
    #                                         self.base_twist[3:])) # last item is base angular velocity
        

    #     if np.isnan(self.state).any():
    #         print('nans in state')
    #         breakpoint()

    #     # Not used in state, but used in _is_terminal() and _reward()    
    #     self.previous_base_twist = self.base_twist
    

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


    
    



    


