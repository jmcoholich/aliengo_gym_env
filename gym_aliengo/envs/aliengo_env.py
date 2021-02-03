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
                    env_mode='pmtg',
                    apply_perturb=True,
                    avg_time_per_perturb=5.0, # seconds
                    max_torque=44.4, # N-m, from URDF
                    kp=0.1, # acts on position error 
                    kd=1.0, # acts on rate of position erorr
                    action_repeat=4,
                    timeout=60.0, # number of seconds to timeout after
                    flat_ground=True, # this is for getting terrain scan in privileged info for Aliengo 
                    realTime=False, # should never be True when training, only for visualzation or debugging MAYBE
                    vis=False,
                    gait_type='trot'):
        # Environment Options
        self.env_mode = env_mode
        self.apply_perturb = apply_perturb
        self.avg_time_per_perturb = avg_time_per_perturb # average time in seconds between perturbations 
        self.n_hold_frames = action_repeat
        self.eps_timeout = 240.0/self.n_hold_frames * timeout # number of steps to timeout after
        # if U[0, 1) is less than this, perturb
        self.perturb_p = 1.0/(self.avg_time_per_perturb * 240.0) * self.n_hold_frames
        self.flat_ground = flat_ground

        # setting these to class variables so that child classes can use it as such. 
        self.realTime = realTime 
        self.max_torque = max_torque
        self.kp = kp 
        self.kd = kd
        self.vis = vis
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
                                            kd=kd,
                                            vis=vis,
                                            gait_type=gait_type)
        self.fake_client = None
        self.client.setGravity(0,0,-9.8)
        if self.realTime:
            warnings.warn('\n\n' + '#'*100 + '\nExternal force/torque disturbances will NOT work properly with '
                            'real time pybullet GUI enabled.\n' + '#'*100 + '\n') # how to make this warning yellow?
        self.client.setRealTimeSimulation(self.realTime) # this has no effect in DIRECT mode, only GUI mode
        self.client.setTimeStep(1/240.0)

        if self.env_mode == 'pmtg':
            self.action_lb, self.action_ub = self.quadruped.get_pmtg_action_bounds() 
            observation_lb, observation_ub = self.quadruped.get_pmtg_observation_bounds()
            self.t = 0.0

        elif self.env_mode == 'hutter_pmtg':
            self.action_lb, self.action_ub = self.quadruped.get_pmtg_action_bounds()
            observation_lb, observation_ub = self.quadruped.get_hutter_pmtg_observation_bounds()

        elif self.env_mode == 'hutter_teacher_pmtg':
            self.action_lb, self.action_ub = self.quadruped.get_pmtg_action_bounds()
            observation_lb, observation_ub = self.quadruped.get_hutter_teacher_pmtg_observation_bounds()

        elif self.env_mode == 'flat':
            self.action_lb, self.action_ub = self.quadruped.get_joint_position_bounds()
            observation_lb, observation_ub = self.quadruped.get_observation_bounds()

        else:
            raise ValueError("env_mode should either be 'pmtg', 'hutter_pmtg', 'hutter_teacher_pmtg', or 'flat'. "
                            "Value {} was given.".format(self.env_mode))

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



    def step(self, action):
        DELTA = 0.01
        if not ((self.action_lb - DELTA <= action) & (action <= self.action_ub + DELTA)).all():
            print("Action passed to env.step(): ", action)
            raise ValueError('Action is out-of-bounds of:\n' + str(self.action_lb) + '\nto\n' + str(self.action_ub)) 

        if self.env_mode in ['pmtg', 'hutter_pmtg', 'hutter_teacher_pmtg'] :
            # f = action[:4]
            f = np.tile(action[0], 4) #TODO decide whether or not to keep this
            residuals = action[4:].reshape((4,3))
            self.quadruped.set_trajectory_parameters(self.t, f=f, residuals=residuals)
            self.t += 1./240. * self.n_hold_frames
        elif self.env_mode == 'flat':
            self.quadruped.set_joint_position_targets(action)
        else: raise ValueError("env_mode should either be 'pmtg', 'hutter_pmtg', 'hutter_teacher_pmtg', or 'flat'. "
                                                                            "Value {} was given.".format(self.env_mode))

        if (np.random.rand() < self.perturb_p) and self.apply_perturb: 
            '''TODO eventually make disturbance generating function that applies disturbances for multiple timesteps'''
            if np.random.rand() > 0.5:
                # TODO returned values will be part of privledged information for teacher training
                force, foot = self.quadruped.apply_foot_disturbance() 
            else:
                # TODO returned values will be part of privledged information for teacher training
                wrench = self.quadruped.apply_torso_disturbance()

        for _ in range(self.n_hold_frames): 
            self.client.stepSimulation()
            if self.vis: self.quadruped.visualize()
        self.eps_step_counter += 1
        self.quadruped.update_state(flat_ground=self.flat_ground, fake_client=self.fake_client)

        if self.env_mode == 'pmtg':
            obs = self.quadruped.get_pmtg_observation()
        elif self.env_mode == 'hutter_pmtg':
            obs = self.quadruped.get_hutter_pmtg_observation()
        elif self.env_mode == 'hutter_teacher_pmtg':
            obs = self.quadruped.get_hutter_teacher_pmtg_observation()
        elif self.env_mode == 'flat':
            obs = self.quadruped.get_observation()
        else: assert False

        info = {}
        done, termination_dict = self._is_state_terminal() # this must come after self._update_state()
        info.update(termination_dict) # termination_dict is an empty dict if not done

        if self.env_mode in ['pmtg', 'hutter_pmtg', 'hutter_teacher_pmtg']:
            rew, rew_dict = self.quadruped.pmtg_reward()
        elif self.env_mode == 'flat':
            raise NotImplementedError
            # rew, rew_dict = self.quadruped.reward()
        else: assert False
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
                self.mean_rew_dict['mean_' + key] = rew_dict[key]
        elif self.eps_step_counter > 1:
            for key in rew_dict:
                self.mean_rew_dict['mean_' + key] += \
                                        (rew_dict[key] - self.mean_rew_dict['mean_' + key])/float(self.eps_step_counter)
        else:
            assert False
            

    def reset(self, base_height=0.48): 
        '''Resets the robot to a neutral standing position, knees slightly bent. The motor control command is to 
        prevent the robot from jumping/falling on first user command. '''

        self.eps_step_counter = 0
        self.client.resetBasePositionAndOrientation(self.quadruped.quadruped,
                                            posObj=[0,0,base_height], 
                                            ornObj=[0,0,0,1.0]) 

        self.quadruped.reset_joint_positions(stochastic=True) 
        for i in range(500): # to let the robot settle on the ground.
            self.client.stepSimulation()
        self.quadruped.update_state(flat_ground=self.flat_ground, fake_client=self.fake_client)
        if self.env_mode == 'pmtg':
            self.t = 0.0
            obs = self.quadruped.get_pmtg_observation()
        elif self.env_mode == 'hutter_pmtg':
            self.t = 0.0
            obs = self.quadruped.get_hutter_pmtg_observation()
        elif self.env_mode == 'hutter_teacher_pmtg':
            self.t = 0.0
            obs = self.quadruped.get_hutter_teacher_pmtg_observation()
        elif self.env_mode == 'flat':
            obs = self.quadruped.get_observation()
        else: assert False

        return obs


    def render(self, mode='human', client=None):
        if client is None:
            client = self.client
        return self.quadruped.render(mode=mode, client=client)


    def close(self):
        '''I belive this is required for an Open AI gym env.'''

        pass


    def _is_state_terminal(self):
        quadruped_done, termination_dict = self.quadruped.is_state_terminal()
        timeout = (self.eps_step_counter >= self.eps_timeout) 
        if timeout:
            termination_dict['TimeLimit.truncated'] = True
        done = quadruped_done or timeout
        return done, termination_dict
            

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


    
    



    


