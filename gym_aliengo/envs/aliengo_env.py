import gym
from gym import error, spaces, utils
from gym.utils import seeding
import sys
import pybullet as p
import os
import time
import numpy as np

class AliengoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=False):
        self._render = render

        if self._render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        urdfFlags = p.URDF_USE_SELF_COLLISION
        self.plane = p.loadURDF(os.path.join(os.path.dirname(__file__), '../urdf/plane.urdf'))
        self.quadruped = p.loadURDF(os.path.join(os.path.dirname(__file__), '../urdf/aliengo.urdf'),
            basePosition=[0,0,0.48],baseOrientation=[0,0,0,1], flags = urdfFlags,useFixedBase=False)

        p.setGravity(0,0,-9.8)
        self.lower_legs = [2,5,8,11]
        for l0 in self.lower_legs:
            for l1 in self.lower_legs:
                if (l1>l0):
                    enableCollision = 1
                    # print("collision for pair",l0,l1, p.getJointInfo(self.quadruped,l0)[12],p.getJointInfo(self.quadruped,l1)[12], "enabled=",enableCollision)
                    p.setCollisionFilterPair(self.quadruped, self.quadruped, l0,l1,enableCollision)

        # p.getCameraImage(480,320)
        p.setRealTimeSimulation(0)

        for i in range (p.getNumJoints(self.quadruped)):
            p.changeDynamics(self.quadruped,i,linearDamping=0, angularDamping=.5)

        self.motor_joint_indices = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16] # the other joints in the urdf are fixed joints 
        self.n_motors = 12
        self.state_space_dim = 12 * 2 + 4 # pos and vel for each motor, plus base orientation (quaternions)
        self.action_space_dim =  12 * 2 
        self.actions_ub = np.empty(self.action_space_dim)
        self.actions_lb = np.empty(self.action_space_dim)

        self.state = np.zeros(self.state_space_dim)
        self.base_position = np.zeros(3) # not returned as observation, but used for other calculations
        self.applied_torques = np.zeros(self.n_motors)
        self.previous_base_twist = np.zeros(6)
        self.previous_lower_limb_vels = np.zeros(4 * 6)
        self.state_noise_std = 0.003125 * 1e-9 * np.array([3.14 * 1e-8, 40* 0.25 * 1e-8] * 12 + [0.78 * 0.125] * 4)


        self._find_action_limits()
        self.num_envs = 1

        self.reward = 0 # this is to store most recent reward
        self.action_space = spaces.Box(
            low=self.actions_lb,
            high=self.actions_ub,
            dtype=np.float32
            )

        self.observation_space = spaces.Box(
            low=np.concatenate((self.actions_lb, -0.78 * np.ones(4))),
            high=np.concatenate((self.actions_ub, 0.78 * np.ones(4))),
            dtype=np.float32
            )


    def step(self, action):
        maxForces = 40 * np.ones(self.n_motors)
        p.setJointMotorControlArray(self.quadruped, self.motor_joint_indices, controlMode=p.POSITION_CONTROL, targetPositions=action[::2], targetVelocities=action[1::2], forces=maxForces)
        p.stepSimulation()
        self._update_state()
        self.reward = self._reward_function()

        if self._is_state_terminal():
            done = True
        else:
            done = False
        info = {'':''} # this is returned so that env.step() matches Open AI gym API
        return self.state + 0 * np.random.normal(scale=self.state_noise_std), self.reward, done, info

        
    def reset(self):

        p.resetBasePositionAndOrientation(self.quadruped, posObj=[0,0,0.48], ornObj=[0,0,0,1])
        for i in self.motor_joint_indices: # for some reason there is no p.resetJointStates
            p.resetJointState(self.quadruped, i, 0, 0)
        self._update_state()


        return self.state


    def render(self, mode='human'):
        '''Setting the render kwarg in the constructor determines if the env will render or not.'''

        # self.render = True
        pass 


    def close(self):
        pass


    def _find_action_limits(self):

       
        for i in range(0, self.n_motors * 2, 2): 
            joint_info = p.getJointInfo(self.quadruped, self.motor_joint_indices[int(i/2)])

            # bounds on joint position
            self.actions_lb[i] = joint_info[8]
            self.actions_ub[i] = joint_info[9]
            
            # bounds on joint velocity
            self.actions_lb[i + 1] = - joint_info[11]
            self.actions_ub[i + 1] =  joint_info[11]

        # no joint limits given for the thigh joints, so set them to plus/minus 90 degrees
        for i in range(self.action_space_dim):
            if self.actions_ub[i] <= self.actions_lb[i]:
                if i%2 == 0: # this is a position state
                    self.actions_lb[i] = -3.14159 * 0
                    self.actions_ub[i] = 3.14159 * 0.5
                else: # this is a velocity (should not reach this using aliengo.urdf)
                    self.actions_lb[i] = -40 
                    self.action_ub[i] = 40

    def _reward_function(self) -> float:
        ''' Calculates reward based off of current state '''

        base_twist = np.array(p.getBaseVelocity(self.quadruped)).flatten()
        base_x_velocity = base_twist[0]
        base_y_velocity = base_twist[1]
        base_accel_penalty = np.power(base_twist[1:] - self.previous_base_twist[1:], 2).mean()
        torque_penalty = np.power(self.applied_torques, 2).mean()
        lower_limb_states = list(p.getLinkStates(self.quadruped, self.lower_legs, computeLinkVelocity=True))
        lower_limb_vels = np.array([lower_limb_states[i][6] + lower_limb_states[i][7] for i in range(4)]).flatten()
        lower_limb_accel_penalty = np.power(lower_limb_vels - self.previous_lower_limb_vels, 2).mean()


        # time.sleep(0.1)

        self.previous_base_twist = base_twist 
        self.previous_lower_limb_vels = lower_limb_vels
        # print(base_x_velocity , 0.0001 * torque_penalty , 0.01 * base_accel_penalty , 0.01 * lower_limb_accel_penalty)
        return base_x_velocity - 0.0001 * torque_penalty - 0.01 * base_accel_penalty - 0.01 * lower_limb_accel_penalty - 0.1 * abs(base_y_velocity)


    def _update_state(self):
        ''' 
        Gets information about state of robot from simulation and fills in the state tensor. 
        The state for the aliengo.urdf (from https://github.com/unitreerobotics/aliengo_pybullet) is the joint[0] position, joint[0] velocity, joint[1] position, ..., joint[11] velocity, then four values for global orientation of base in quaternions.
        The order of the joint groups are FR (hip, thigh, calf), FL (hip, thigh, calf), RR then RL. 
        This function also updates the applied joint torques that are penalized in the _reward_function.

        '''

        joint_states = list(p.getJointStates(self.quadruped, self.motor_joint_indices))
        self.applied_torques = np.array([joint_states[i][3] for i in range(self.n_motors)])

        for i in range(0, self.n_motors * 2, 2):
            self.state[i] = joint_states[int(i/2)][0]
            self.state[i+1] = joint_states[int(i/2)][1]


        # base_state = p.getLinkState(self.quadruped, 0, computeLinkVelocity=0) 
        # self.state[i+2], self.state[i+3], self.state[i+4] = base_state[0]  
        # self.state[i+5], self.state[i+6], self.state[i+7], self.state[i+8] = base_state[1]

        base_position_and_orientation = p.getBasePositionAndOrientation(self.quadruped)
        self.state[i+2], self.state[i+3], self.state[i+4], self.state[i+5] = base_position_and_orientation[1]

        self.base_position = list(base_position_and_orientation[0])

        if np.isnan(self.state).any():
            print('nans in state')
            breakpoint()


    def _is_state_terminal(self) -> bool:
        ''' Calculates whether to end current episode due to failure based on current state. Does not consider timeout '''

        base_z_position = self.base_position[2]
        height_out_of_bounds = (base_z_position < 0.23) or (base_z_position > 0.8)
        falling = (abs(np.array(list(p.getEulerFromQuaternion(self.state[-4:])))) > 0.78).any() # 0.78 rad is 45 deg
        return falling or height_out_of_bounds

