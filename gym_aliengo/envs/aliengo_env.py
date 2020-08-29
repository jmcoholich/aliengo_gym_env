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
        self._apply_perturbations = False
        self._is_render = render

        if self._is_render:
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
        self.state_space_dim = 12 * 3 + 4 # torque, pos, and vel for each motor, plus base orientation (quaternions)
        self.action_space_dim =  12 
        self.actions_ub = np.empty(self.action_space_dim)
        self.actions_lb = np.empty(self.action_space_dim)

        self.state = np.zeros(self.state_space_dim)
        self.base_position = np.zeros(3) # not returned as observation, but used for other calculations
        self.applied_torques = np.zeros(self.n_motors)
        self.joint_velocities = np.zeros(self.n_motors)
        self.joint_positions = np.zeros(self.n_motors)
        self.base_orientation = np.zeros(4)
        self.previous_base_twist = np.zeros(6)
        self.previous_lower_limb_vels = np.zeros(4 * 6)
        # self.state_noise_std = 0.03125  * np.array([3.14, 40] * 12 + [0.78 * 0.25] * 4 + [0.25] * 3)
        self.perturbation_rate = 0.01 # probability that a random perturbation is applied to the torso
        self.max_torque = 40
        self.kp = 1.0
        self.kd = 0.02


        self._find_space_limits()
        self.num_envs = 1

        self.reward = 0 # this is to store most recent reward
        self.action_space = spaces.Box(
            low=self.actions_lb,
            high=self.actions_ub,
            dtype=np.float32
            )

        self.observation_space = spaces.Box(
            low=np.concatenate((-self.max_torque * np.ones(12), self.actions_lb, -40 * np.ones(12), -0.78 * np.ones(4))),
            high=np.concatenate((self.max_torque * np.ones(12), self.actions_ub, 40 * np.ones(12), 0.78 * np.ones(4))),
            dtype=np.float32
            )


    def step(self, action):

        # action = np.clip(action, self.action_space.low, self.action_space.high)
        if not ((self.action_space.low <= action) & (action <= self.action_space.high)).all():
            raise ValueError('Action is out-of-bounds') 


        p.setJointMotorControlArray(self.quadruped,
            self.motor_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=action,
            forces=self.max_torque * np.ones(self.n_motors),
            positionGains=self.kp * np.ones(12),
            velocityGains=self.kd * np.ones(12))

        if (np.random.rand() > self.perturbation_rate) and self._apply_perturbations: 
            self._apply_perturbation()
        p.stepSimulation()
        self._update_state()
        self.reward = self._reward_function()

        if self._is_state_terminal():
            done = True
        else:
            done = False
        info = {'':''} # this is returned so that env.step() matches Open AI gym API
        return self.state, self.reward, done, info

    
    def _apply_perturbation(self):
        if np.random.rand() > 0.5: # apply force
            force = tuple(10 * (np.random.rand(3) - 0.5))
            p.applyExternalForce(self.quadruped, -1, force, (0,0,0), p.LINK_FRAME)
        else: # apply torque
            torque = tuple(0.5 * (np.random.rand(3) - 0.5))
            p.applyExternalTorque(self.quadruped, -1, torque, p.LINK_FRAME)


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


    def _find_space_limits(self):

       
        for i in range(self.n_motors): 

            joint_info = p.getJointInfo(self.quadruped, self.motor_joint_indices[i])
            
            # bounds on joint position
            self.actions_lb[i] = joint_info[8]
            self.actions_ub[i] = joint_info[9]
            
            # bounds on joint velocity
            # self.actions_lb[i + 12] = - joint_info[11]
            # self.actions_ub[i + 12] =  joint_info[11]

        # no joint limits given for the thigh joints, so set them to plus/minus 90 degrees
        for i in range(self.action_space_dim):
            if self.actions_ub[i] <= self.actions_lb[i]:
                assert i < 12 # this is a position state
                self.actions_lb[i] = -3.14159 * 0
                self.actions_ub[i] = 3.14159 * 0.5
                # else: # this is a velocity (should not reach this using aliengo.urdf)
                #     assert False
                #     # self.actions_lb[i] = -40 
                #     # self.action_ub[i] = 40

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

        # lower_limb_height_bonus = np.array([lower_limb_states[i][0][2] for i in range(4)]).mean()

        # time.sleep(0.1)

        self.previous_base_twist = base_twist 
        self.previous_lower_limb_vels = lower_limb_vels
        # print(base_x_velocity , 0.0001 * torque_penalty , 0.01 * base_accel_penalty , 0.01 * lower_limb_accel_penalty, 0.1 * lower_limb_height_bonus)
        return 1.0*base_x_velocity - 0.0001 * torque_penalty #- 0.01 * base_accel_penalty \
             # - 0.01 * lower_limb_accel_penalty - 0.1 * abs(base_y_velocity) # \
             # + 0.1 * lower_limb_height_bonus


    def _update_state(self):

        joint_states = list(p.getJointStates(self.quadruped, self.motor_joint_indices))
        self.applied_torques  = np.array([joint_states[i][3] for i in range(self.n_motors)])
        self.joint_positions  = np.array([joint_states[i][0] for i in range(self.n_motors)])
        self.joint_velocities = np.array([joint_states[i][1] for i in range(self.n_motors)])

        base_position, base_orientation = p.getBasePositionAndOrientation(self.quadruped)

        self.base_position = np.array(base_position)
        self.base_orientation = np.array(base_orientation)

        self.state = np.concatenate((self.applied_torques, self.joint_positions, self.joint_velocities, self.base_orientation))

        if np.isnan(self.state).any():
            print('nans in state')
            breakpoint()


    def _is_state_terminal(self) -> bool:
        ''' Calculates whether to end current episode due to failure based on current state. Does not consider timeout '''

        base_z_position = self.base_position[2]
        height_out_of_bounds = (base_z_position < 0.23) or (base_z_position > 0.8)
        falling = (abs(np.array(list(p.getEulerFromQuaternion(self.base_orientation)))) > 0.78).any() # 0.78 rad is 45 deg
        return falling or height_out_of_bounds

