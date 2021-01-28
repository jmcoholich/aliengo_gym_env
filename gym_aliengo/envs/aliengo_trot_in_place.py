
from gym_aliengo.envs import aliengo_env
from pybullet_utils import bullet_client as bc
import pybullet as p
from os.path import dirname, join
import numpy as np

'''
This class is the same as Aliengo-v0, but with the following changes:
- reward function excludes term for forward motion. All other terms included
- the mode is force to be teacher_PMTG
- the action bounds are different
- the frequency set for the first leg, is set for all the legs.
'''
class AliengoTrotInPlace(aliengo_env.AliengoEnv):
    def __init__(self, **kwargs):
        kwargs['env_mode'] = 'hutter_teacher_pmtg' 
        super().__init__(**kwargs)
        self.action_lb = np.array([-1.0] * 4 + [-0.2, -0.2, -0.2] * 4)
        self.action_ub = np.array([3.0] * 4 + [0.2, 0.2, 0.2] * 4) 

    

    # this is copied and pasted 
    def step(self, action):
        assert self.env_mode == 'hutter_teacher_pmtg'
        DELTA = 0.01
        if not ((self.action_lb - DELTA <= action) & (action <= self.action_ub + DELTA)).all():
            print("Action passed to env.step(): ", action)
            raise ValueError('Action is out-of-bounds of:\n' + str(self.action_lb) + '\nto\n' + str(self.action_ub)) 

        if self.env_mode in ['pmtg', 'hutter_pmtg', 'hutter_teacher_pmtg'] :
            f = np.tile(action[0], 4)
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

        for _ in range(self.n_hold_frames): self.client.stepSimulation()
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
            rew, rew_dict = self.quadruped.trot_in_place_reward()
        elif self.env_mode == 'flat':
            raise NotImplementedError
            # rew, rew_dict = self.quadruped.reward()
        else: assert False
        self._update_mean_rew_dict(rew_dict)

        if done:
            info['distance_traveled']   = self.quadruped.base_position[0]
            info.update(self.mean_rew_dict)

        return obs, rew, done, info


