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
main.py --env-name gym_aliengo:FootstepParam-v0 --wandb-project FootstepParam-v0 --recurrent-policy --num-processes 40 --num-steps 300 --seed 3051466742 --entropy-coef 0.01 --gpu-idx 2


Termination: same as parent

"""

class FootstepParam(aliengo_env.AliengoEnv):

    def __init__(self, num_footstep_cycles=10, **kwargs):    
        super().__init__(**kwargs)

        self.step_len = 0.13
        
        if self.vis:
            self._init_vis()

        self.current_footstep = 0
        self.num_footstep_cycles = num_footstep_cycles
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


    def _is_state_terminal(self): 
        done, termination_dict = super()._is_state_terminal()
        # end on penultimate footstep, to avoid indexing errors
        reached_end = self.current_footstep == (self.num_footstep_cycles * 4 - 2)
        done = done or reached_end
        if reached_end: # this is effectively same as timeout, for reward purposes
            termination_dict['TimeLimit.truncated'] = True
        return done, termination_dict


    def _init_vis(self):
        shape = self.client.createVisualShape(p.GEOM_CYLINDER, 
                                            radius=.06, 
                                            length=.001, 
                                            rgbaColor=[191/255., 87/255., 0, 0.95])
        self.foot_step_marker = self.client.createMultiBody(baseVisualShapeIndex=shape)


    def generate_footstep_locations(self):
        ''' This is just a straight path on flat ground for now '''
        
        self.footsteps = np.zeros((4, 3)) # each footstep is an x and y position
        step_len = self.step_len + (np.random.random_sample() - 0.5) * 0.02
        # step_len = 1.0
        width = 0.25 + (np.random.random_sample() - 0.5) * 0.01
        length = 0.45 
        len_offset = -0.02

        if np.random.random_sample() > 0.5:
            self.footstep_idcs = [2,0,3,1]
            self.footsteps[0] = np.array([-length/2.0 + len_offset + step_len, -width/2.0, 0]) # RR
            self.footsteps[1] = np.array([length/2.0  + len_offset + step_len, -width/2.0, 0]) # FR
            self.footsteps[2] = np.array([-length/2.0 + len_offset + 2 * step_len, width/2.0, 0]) # RL
            self.footsteps[3] = np.array([length/2.0  + len_offset + 2 * step_len, width/2.0, 0]) # FL
        else:
            self.footstep_idcs = [3,1,2,0]
            self.footsteps[0] = np.array([-length/2.0 + len_offset + step_len, width/2.0, 0]) # RL
            self.footsteps[1] = np.array([length/2.0  + len_offset + step_len, width/2.0, 0]) # FL
            self.footsteps[2] = np.array([-length/2.0 + len_offset + 2 * step_len, -width/2.0, 0]) # RR
            self.footsteps[3] = np.array([length/2.0  + len_offset + 2 * step_len, -width/2.0, 0]) # FR
        
        self.footsteps = np.tile(self.footsteps, (self.num_footstep_cycles, 1))
        self.footsteps[:, 0] += np.arange(self.num_footstep_cycles).repeat(4) * step_len * 2
        # for i in range(self.num_footstep_cycles):
        #     self.footsteps[i*4: (i+1)*4, 0] += i * 2 * step_len 
        # if np.random.random_sample() > 0.5: # swap which rear foot goes first, so agent doesn't memorize first foot.
        # if True: # swap which rear foot goes first, so agent doesn't memorize first foot.
        #     # idx = np.arange(self.num_footstep_cycles * 2) * 2
        #     idx = np.arange(self.num_footstep_cycles * 2) + np.arange(self.num_footstep_cycles * 1).repeat(2) * 2
        #     self.footsteps[np.concatenate((idx, idx + 2))] = self.footsteps[np.concatenate((idx + 2, idx))]
        if self.vis:
            self.client.resetBasePositionAndOrientation(self.foot_step_marker,
                                                        self.footsteps[self.current_footstep], 
                                                        [0, 0, 0, 1])


    # def get_current_foot_idx(self):
    #     if self.current_footstep%4 == 0: # RR
    #         return 2
    #     elif self.current_footstep%4 == 1: # FR
    #         return 0
    #     elif self.current_footstep%4 == 2: # RL
    #         return 3
    #     elif self.current_footstep%4 == 3: # FL
    #         return 1
    #     else:
    #         assert False


    def get_current_foot_global_pos(self):
        """Returns position of the bottom of the foot."""

        foot = self.footstep_idcs[self.current_footstep%4]
        pos = np.array(self.client.getLinkState(self.quadruped.quadruped, self.quadruped.foot_links[foot])[0])
        pos[2] -= 0.0265
        return pos

    
    def calc_curr_foostep_dist(self):
        global_pos = self.get_current_foot_global_pos()
        return np.linalg.norm(global_pos - self.footsteps[self.current_footstep])


    # def footstep_rew(self):
        
    #     tol = 0.03 # in meters
    #     # find current foot
    #     # FR, FL, RR, RL

    #     dist = self.calc_curr_foostep_dist()

    #     if (dist < tol) and (self.quadruped.get_foot_contacts()[self.footstep_idcs[self.current_footstep%4]] > 10.0): 
    #         self.current_footstep += 1
    #         rew = 1.0
    #         if self.vis: print("#" * 100 + '\n' + 'footstep reached' + '\n' + '#' * 100)
    #     else:
    #         rew = -dist
        
    #     if self.vis:
    #         print('Footstep rew: {:.2f}'.format(rew))
    #         self.client.resetBasePositionAndOrientation(self.foot_step_marker,
    #                                                     self.footsteps[self.current_footstep], 
    #                                                     [0, 0, 0, 1])
    #     return rew


    def footstep_vel_rew(self): 
        
        tol = 0.03 # in meters
        max_rewarded_speed = 1.0 # m/s

        curr_dist = self.calc_curr_foostep_dist()

        if (curr_dist < tol) and \
                            (self.quadruped.get_foot_contacts()[self.footstep_idcs[self.current_footstep%4]] > 10.0): 
            self.current_footstep += 1
            rew = 3.0
            if self.vis: print("#" * 100 + '\n' + 'footstep reached' + '\n' + '#' * 100)
            self.prev_dist = self.calc_curr_foostep_dist() # will return a different value, since current_footstep ++
        else:
            rew = np.clip((self.prev_dist - curr_dist)/ (1.0/240 * self.n_hold_frames), -np.inf, max_rewarded_speed)
            self.prev_dist = curr_dist
        
        if self.vis:
            print('Footstep vel rew: {:.5f}'.format(rew))
            self.client.resetBasePositionAndOrientation(self.foot_step_marker,
                                                        self.footsteps[self.current_footstep], 
                                                        [0, 0, 0, 1])
        return rew


    def footstep_stay_rew(self):
        """Reward the foot opposite in cycle to be planted on the ground."""

        contact = self.quadruped.get_foot_contacts()[self.footstep_idcs[(self.current_footstep + 2)%4]] > 10.0
        return 1.0 * contact


    
    def reward(self):
        '''Get rid of rewards for fwd motion.'''
        
        base_vel, base_avel = self.client.getBaseVelocity(self.quadruped.quadruped)

        base_motion_rew = np.exp(-1.5 * (base_vel[1] * base_vel[1])) + \
                                            np.exp(-1.5 * (base_avel[0] * base_avel[0] + base_avel[1] * base_avel[1]))

        # foot_clearance_rew = self._foot_clearance_rew()

        body_collision_rew = -(self.quadruped.is_non_foot_ground_contact() + self.quadruped.self_collision())

        target_smoothness_rew = - np.linalg.norm(self.quadruped.true_joint_position_target_history[0] \
                                                - 2 * self.quadruped.true_joint_position_target_history[1] + \
                                                self.quadruped.true_joint_position_target_history[2])

        torque_rew = -np.linalg.norm(self.quadruped.applied_torques, 1)

        # footstep_rew = self.footstep_rew()
        footstep_vel_rew = self.footstep_vel_rew()
        footstep_stay_rew = self.footstep_stay_rew()

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
                    # 'foostep_rew':footstep_rew}
                    'foostep_vel_rew':footstep_vel_rew,
                    'footstep_stay_rew':footstep_stay_rew}
        

        # other stuff to track
        rew_dict['x_vel'] = self.quadruped.base_vel[0]

        total_rew = 0.10 * base_motion_rew + 0.20 * body_collision_rew + 0.10 * target_smoothness_rew \
                    + 2e-5 * torque_rew + 1.0 * footstep_vel_rew + 0.5 * footstep_stay_rew #0.1 * footstep_rew

        return total_rew, rew_dict


    def step(self, action):
        DELTA = 0.01
        if not ((self.action_lb - DELTA <= action) & (action <= self.action_ub + DELTA)).all():
            print("Action passed to env.step(): ", action)
            raise ValueError('Action is out-of-bounds of:\n' + str(self.action_lb) + '\nto\n' + str(self.action_ub)) 
        
        self.quadruped.footstep_param_action(action)
        rand = np.random.random_sample()
        p = 0.005
        if rand < p:
            if rand < p / 2.0: 
                self.quadruped.apply_foot_disturbance(self, force=None, foot=None, max_force_mag=2500 * 0.1)
            else:
                self.quadruped.apply_torso_disturbance(self, wrench=None, max_force_mag=5000 * 0.1, \
                                                                                            max_torque_mag=500 * 0.1)

        for _ in range(self.n_hold_frames): 
            self.client.stepSimulation()
            if self.vis: self.quadruped.visualize()
        self.eps_step_counter += 1
        self.quadruped.update_state(flat_ground=self.flat_ground, fake_client=self.fake_client, update_priv_info=False)

        obs = self.get_obs()

        info = {}
        done, termination_dict = self._is_state_terminal() # this must come after self._update_state()
        info.update(termination_dict) # termination_dict is an empty dict if not done

        rew, rew_dict = self.reward()
        self.update_mean_rew_dict(rew_dict)

        if done:
            info['distance_traveled'] = self.quadruped.base_position[0]
            info['footstep_reached'] = self.current_footstep
            info.update(self.mean_rew_dict)

        return obs, rew, done, info


    def reset(self, base_height=0.48, stochastic=True): 
        '''Resets the robot to a neutral standing position, knees slightly bent. The motor control command is to 
        prevent the robot from jumping/falling on first user command. '''

        if self.vis: print('*' * 100 + '\n' + 'Resetting' + '\n' + '*' * 100)
        self.current_footstep = 0
        self.eps_step_counter = 0
        self.generate_footstep_locations()
        self.client.resetBasePositionAndOrientation(self.quadruped.quadruped,
                                            posObj=[0,0,base_height], 
                                            ornObj=[0,0,0,1.0]) 

        self.quadruped.reset_joint_positions(stochastic=stochastic) 
        for i in range(500): # to let the robot settle on the ground.
            self.client.stepSimulation()
        self.quadruped.update_state(flat_ground=self.flat_ground, fake_client=self.fake_client, update_priv_info=False)
        self.prev_dist = self.calc_curr_foostep_dist()
        obs = self.get_obs()

        return obs


    def get_obs(self):
        obs = np.concatenate((self.quadruped.footstep_param_obs(), 
                            np.array([self.footstep_idcs[self.current_footstep%4]]), 
                            self.get_current_foot_global_pos() - self.footsteps[self.current_footstep]))
        return obs
    

def render_all_footsteps(env):
    for i in range(int(len(env.footsteps)/4)):
        shape = env.client.createVisualShape(p.GEOM_CYLINDER, 
                                            radius=.06, 
                                            length=.001, 
                                            rgbaColor=[*np.random.random_sample(3),0.95])
        for j in range(4):
            env.client.createMultiBody(baseVisualShapeIndex=shape, basePosition=env.footsteps[i*4 + j])
            env.client.addUserDebugText(str(j), env.footsteps[i*4 + j])
    # while True:
    #     time.sleep(10)

 
if __name__ == '__main__':
    env = FootstepParam(render=True, vis=True, fixed=True)
    env.reset(stochastic=False)
    render_all_footsteps(env)
    while True:
        env.client.stepSimulation()
        env.reward()
        time.sleep(1/240. * 4)

    # foot_positions = np.zeros((4, 3))
    # lateral_offset = 0.11
    # foot_positions[:,-1] = -0.4
    # foot_positions[[0,2], 1] = -lateral_offset
    # foot_positions[[1,3], 1] = lateral_offset
    # action =  np.concatenate((foot_positions.flatten(), 
    #                     np.zeros(12)))# first 12 is footstep xyz position, last 12 is added joint positions
    #                     # env.quadruped._positions_to_actions(np.zeros(12))))# first 12 is footstep xyz position, last 12 is added joint positions
    # # action = env.action_space.high
    # # action[12:] = 0.0
    # while True:
    #     env.step(action)
    #     # env.quadruped.set_foot_positions(foot_positions)
    #     # env.client.stepSimulation()
    #     time.sleep(1/240.0 * 4)
