from gym_aliengo.envs import aliengo_env
from pybullet_utils import bullet_client as bc
import pybullet as p
from os.path import dirname, join
import numpy as np
import time
import gym
from gym import error, spaces, utils

"""
So here I will only change the reward function. Instead of a forward reward, I will just give footsteps leading forward.
Also, the action space will just be foot positions, I can get rid of the parametrization.
 - lets do this first. 

Or keep the parametrization and have the rewarded foot position change each phase cycle.

Then later I can build a vision part, that will just learn to output this sequence of footsteps. It can given a 
hand-crafted costmap (plus the footsteps can't be too far apart) for reward training at first, 
then later I can train it based on the success of a well-trained quadruped agent.


Train this agent with 
main.py --env-name gym_aliengo:FootstepParam-v0 --wandb-project FootstepParam-v0 --recurrent-policy --num-processes 40 --num-steps 300 --seed 3051466742 --entropy-coef 0.02 --gpu-idx 2


Termination: same as parent

"""

class FootstepParam(aliengo_env.AliengoEnv):

    def __init__(self, **kwargs):        
        super().__init__(**kwargs)
        if self.vis:
            self._init_vis()

        self.current_footstep = 0
        self.generate_footstep_locations()

        self.action_lb, self.action_ub = self.quadruped.footstep_param_action_bounds()
        self.action_space = spaces.Box(
            low=self.action_lb,
            high=self.action_ub,
            dtype=np.float32
            )

        observation_lb, observation_ub = self.quadruped.footstep_param_obs_bounds()
        self.observation_space = spaces.Box(
            low=observation_lb,
            high=observation_ub,
            dtype=np.float32
            )




    def _init_vis(self):
        shape = self.client.createVisualShape(p.GEOM_CYLINDER, 
                                            radius=.06, 
                                            length=.001, 
                                            rgbaColor=[191/255.,87./255,0,0.95])
        self.foot_step_marker = self.client.createMultiBody(baseVisualShapeIndex=shape)


    def generate_footstep_locations(self):
        ''' This is just a straight path on flat ground for now '''
        
        self.footsteps = np.zeros((4, 3)) # each footstep is an x and y position
        step_len = 0.15 + (np.random.random_sample() - 0.5) * 0.02
        width = 0.25 + (np.random.random_sample() - 0.5) * 0.01
        length = 0.45 
        len_offset = -0.02
        self.footsteps[0] = np.array([-length/2.0 + len_offset + step_len, -width/2.0, 0]) # RR
        self.footsteps[1] = np.array([length/2.0 + len_offset + step_len, -width/2.0, 0]) # FR
        self.footsteps[2] = np.array([-length/2.0 + len_offset + 2 * step_len, width/2.0, 0]) # RL
        self.footsteps[3] = np.array([length/2.0 + len_offset + 2 * step_len, width/2.0, 0]) # FL
        
        n_cycles = 5

        self.footsteps = np.tile(self.footsteps, (n_cycles, 1))
        for i in range(n_cycles):
            self.footsteps[i*4: (i+1)*4, 0] += i * 2 * step_len 
        
        if self.vis:
            self.client.resetBasePositionAndOrientation(self.foot_step_marker,
                                                        self.footsteps[self.current_footstep], 
                                                        [0, 0, 0, 1])


    def get_current_foot_global_pos(self):
        if self.current_footstep%4 == 0: # RR
            foot = 2
        elif self.current_footstep%4 == 1: # FR
            foot = 0
        elif self.current_footstep%4 == 2: # RL
            foot = 3
        elif self.current_footstep%4 == 3: # FL
            foot = 1
        else:
            assert False

        # get foot positions
        return self.client.getLinkState(self.quadruped.quadruped, self.quadruped.foot_links[foot])[0]


    
    def footstep_rew(self):
        
        tol = 0.03 # in meters
        # find current foot
        # FR, FL, RR, RL
        global_pos = self.get_current_foot_global_pos()

        dist = np.linalg.norm(global_pos - self.footsteps[self.current_footstep])
        if dist < tol: 
            self.current_footstep += 1
            rew = 1.0
        else:
            rew = -dist
        return rew


    
    def reward(self):
        '''Get rid of rewards for fwd motion.'''
        
        # speed_treshold = 0.5 # m/s
        base_vel, base_avel = self.client.getBaseVelocity(self.quadruped.quadruped)
        # lin_vel_rew = np.exp(-2.0 * (base_vel[0] - speed_treshold) * (base_vel[0] - speed_treshold)) \
        #                                                                         if base_vel[0] < speed_treshold else 1.0

        # # give reward if we are pointed the right direction
        # _, _, yaw = self.client.getEulerFromQuaternion(self.base_orientation)
        # angular_rew = np.exp(-1.5 * abs(yaw)) # if yaw is zero this is one. 

        base_motion_rew = np.exp(-1.5 * (base_vel[1] * base_vel[1])) + \
                                            np.exp(-1.5 * (base_avel[0] * base_avel[0] + base_avel[1] * base_avel[1]))

        # foot_clearance_rew = self._foot_clearance_rew()

        body_collision_rew = -(self.quadruped.is_non_foot_ground_contact() + self.quadruped.self_collision())

        target_smoothness_rew = - np.linalg.norm(self.quadruped.true_joint_position_target_history[0] \
                                                - 2 * self.quadruped.true_joint_position_target_history[1] + \
                                                self.quadruped.true_joint_position_target_history[2])

        torque_rew = -np.linalg.norm(self.quadruped.applied_torques, 1)

        footstep_rew = self.footstep_rew()

        # knee_force_rew = -np.abs(self.reaction_forces[[[2],[5],[8],[11]],[[1,3,5]]]).sum()
        # knee_force_ratio_rew = np.abs(self.reaction_forces[[[2],[5],[8],[11]],[[0,2,4]]]).sum() /\
        #                                                 np.abs(self.reaction_forces[[[2],[5],[8],[11]],[[1,3,5]]]).sum()

        # wide_step_rew = self._wide_step_rew()

        # rew_dict includes all the things I want to keep track of an average over an entire episode, to be logged
        # add terms of reward function
        rew_dict = {'base_motion_rew': base_motion_rew, 
                    'body_collision_rew':body_collision_rew, 
                    'target_smoothness_rew':target_smoothness_rew,
                    'torque_rew':torque_rew,
                    'foostep_rew':footstep_rew}

        # other stuff to track
        rew_dict['x_vel'] = self.quadruped.base_vel[0]

        total_rew = 0.10 * base_motion_rew + 0.20 * body_collision_rew + 0.10 * target_smoothness_rew \
                    + 2e-5 * torque_rew + 1.0 * footstep_rew
        return total_rew, rew_dict


    def step(self, action):
        DELTA = 0.01
        if not ((self.action_lb - DELTA <= action) & (action <= self.action_ub + DELTA)).all():
            print("Action passed to env.step(): ", action)
            raise ValueError('Action is out-of-bounds of:\n' + str(self.action_lb) + '\nto\n' + str(self.action_ub)) 

        # if self.env_mode in ['pmtg', 'hutter_pmtg', 'hutter_teacher_pmtg'] :
        #     # f = action[:4]
        #     f = np.tile(action[0], 4) #TODO decide whether or not to keep this
        #     residuals = action[4:].reshape((4,3))
        #     self.quadruped.set_trajectory_parameters(self.t, f=f, residuals=residuals)
        #     self.t += 1./240. * self.n_hold_frames
        # elif self.env_mode == 'flat':
        #     self.quadruped.set_joint_position_targets(action)
        # else: raise ValueError("env_mode should either be 'pmtg', 'hutter_pmtg', 'hutter_teacher_pmtg', or 'flat'. "
        #                                                                     "Value {} was given.".format(self.env_mode))
        self.quadruped.footstep_param_action(action)

        # if (np.random.rand() < self.perturb_p) and self.apply_perturb: 
        #     '''TODO eventually make disturbance generating function that applies disturbances for multiple timesteps'''
        #     if np.random.rand() > 0.5:
        #         # TODO returned values will be part of privledged information for teacher training
        #         force, foot = self.quadruped.apply_foot_disturbance() 
        #     else:
        #         # TODO returned values will be part of privledged information for teacher training
        #         wrench = self.quadruped.apply_torso_disturbance()

        for _ in range(self.n_hold_frames): 
            self.client.stepSimulation()
            if self.vis: self.quadruped.visualize()
        self.eps_step_counter += 1
        self.quadruped.update_state(flat_ground=self.flat_ground, fake_client=self.fake_client, update_priv_info=False)

        # if self.env_mode == 'pmtg':
        #     obs = self.quadruped.get_pmtg_observation()
        # elif self.env_mode == 'hutter_pmtg':
        #     obs = self.quadruped.get_hutter_pmtg_observation()
        # elif self.env_mode == 'hutter_teacher_pmtg':
        #     obs = self.quadruped.get_hutter_teacher_pmtg_observation()
        # elif self.env_mode == 'flat':
        #     obs = self.quadruped.get_observation()
        # else: assert False
        obs = self.get_obs()

        info = {}
        done, termination_dict = self._is_state_terminal() # this must come after self._update_state()
        info.update(termination_dict) # termination_dict is an empty dict if not done

        # if self.env_mode in ['pmtg', 'hutter_pmtg', 'hutter_teacher_pmtg']:
        #     rew, rew_dict = self.quadruped.pmtg_reward()
        # elif self.env_mode == 'flat':
        #     raise NotImplementedError
        #     # rew, rew_dict = self.quadruped.reward()
        # else: assert False
        rew, rew_dict = self.reward()
        self.update_mean_rew_dict(rew_dict)

        if done:
            info['distance_traveled'] = self.quadruped.base_position[0]
            info.update(self.mean_rew_dict)

        return obs, rew, done, info


    def reset(self, base_height=0.48, stochastic=True): 
        '''Resets the robot to a neutral standing position, knees slightly bent. The motor control command is to 
        prevent the robot from jumping/falling on first user command. '''

        self.eps_step_counter = 0
        self.client.resetBasePositionAndOrientation(self.quadruped.quadruped,
                                            posObj=[0,0,base_height], 
                                            ornObj=[0,0,0,1.0]) 

        self.quadruped.reset_joint_positions(stochastic=stochastic) 
        for i in range(500): # to let the robot settle on the ground.
            self.client.stepSimulation()
        self.quadruped.update_state(flat_ground=self.flat_ground, fake_client=self.fake_client, update_priv_info=False)
        # if self.env_mode == 'pmtg':
        #     self.t = 0.0
        #     obs = self.quadruped.get_pmtg_observation()
        # elif self.env_mode == 'hutter_pmtg':
        #     self.t = 0.0
        #     obs = self.quadruped.get_hutter_pmtg_observation()
        # elif self.env_mode == 'hutter_teacher_pmtg':
        #     self.t = 0.0
        #     obs = self.quadruped.get_hutter_teacher_pmtg_observation()
        # elif self.env_mode == 'flat':
        #     obs = self.quadruped.get_observation()
        # else: assert False
        obs = self.get_obs()

        return obs


    def get_obs(self):
        obs = np.concatenate((self.quadruped.footstep_param_obs(), 
                            np.array([self.current_footstep%4]), 
                            self.get_current_foot_global_pos() - self.footsteps[self.current_footstep]))
        return obs
    

if __name__ == '__main__':
    env = FootstepParam(render=True, vis=True)
    env.reset(stochastic=False)
    # TODO make sure there are no randomization or disturbances, since I'm giving it a completely deterministic footstep
    # sequence
    # env.quadruped.visualize()
    # env.generate_footstep_locations()
    # env.step(np.zeros_like(env.action_space.high))
    while True:
        time.sleep(10)
