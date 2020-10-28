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

'''Agent can be trained with PPO algorithm from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail. Run:
 python main.py --env-name "gym_aliengo:aliengo-v0" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 10 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 20 --num-mini-batch 4 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 10000000 --use-linear-lr-decay=True --use-proper-time-limits

default params with 10x samples:
 python main.py --env-name "gym_aliengo:aliengo-v0" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 10 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 10000000 --use-linear-lr-decay=True --use-proper-time-limits --save-dir ./trained_models/test4 --seed 4

 python main.py --env-name "gym_aliengo:aliengo-v0" --algo ppo --use-gae --log-interval 1 --num-steps 6000 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 20 --num-mini-batch 4 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 10_000_000 --use-linear-lr-decay=True --use-proper-time-limits --save-dir ./trained_models/test9 --seed 9


Things to try
- plot entropy (I know the PPO kostrikov is logging lots of stuff), and plot other stuff too. Try wandb for this. Also use tmux lol.
- higher penalty on torques


Things I have tried
- get simulation GUI on my laptop and probe it so I know pm 1 produces desired range of motion 
- hyperparam sweep
- normalize action space to fall within pm 1
- changing max torque available 
- making sure my changes to avoid jumping on startup were realized (pip install -e .)
- printing reward function to video
- letting train longer
- made sure its not always using max force
- fixed issue where I didn't specify the PhysicsclientID


Things that might be wrong with the code
- are the limits of the observation space used for anything? How do I know the joint vel limit is 40? (plus maybe the 
quaternion orientation limit is bad)
- should my position observations also be normalized? (I don't think so)
- make sure I pip install -e .
- I halfed the euler angle for the episode to terminate from robot falling
- Perhaps I need to add base velocity to the code? I don't think the policy net can calculate translational velocity
 if it is not touching the floor
 
 Allocate and invest money and check amazon card. Make sure payment I made to bursar was correct. 

 Perhaps I should take the norm or sum of the foot contact forces. Do some investigation.


JOINTS ARE GOING OUT OF BOUNDS AGAIN and its still not quite good. Why doen't it learn to just keep its feet under it? 
Perhaps add a reward just for existing.
 '''
class AliengoEnv(gym.Env):

    def __init__(self, render=False):
        self._apply_perturbations = False
        self._is_render = render

        if self._is_render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        if self.client == -1:
            raise RuntimeError('Pybullet could not connect to physics client')

        urdfFlags = p.URDF_USE_SELF_COLLISION
        self.plane = p.loadURDF(os.path.join(os.path.dirname(__file__), '../urdf/plane.urdf'), 
                                physicsClientId=self.client)
        self.quadruped = p.loadURDF(os.path.join(os.path.dirname(__file__), '../urdf/aliengo.urdf'),
                                    basePosition=[0,0,0.48], 
                                    baseOrientation=[0,0,0,1], 
                                    flags = urdfFlags, 
                                    useFixedBase=False,
                                    physicsClientId=self.client)

        # fixed base for debugging 
        # self.quadruped = p.loadURDF(os.path.join(os.path.dirname(__file__), '../urdf/aliengo.urdf'),
        #     basePosition=[0,0,1.0],baseOrientation=[0,0,0,1], flags = urdfFlags,useFixedBase=True, 
        #       physicsClientId=self.client)

        p.setGravity(0,0,-9.8, physicsClientId=self.client)
        self.foot_links = [5, 9, 13, 17]
        # self.lower_legs = [2,5,8,11]
        # for l0 in self.lower_legs:
        #     for l1 in self.lower_legs:
        #         if (l1>l0):
        #             enableCollision = 1
        #             # print("collision for pair",l0,l1, p.getJointInfo(self.quadruped,l0, physicsClientId=self.client)[12],p.getJointInfo(self.quadruped,l1, physicsClientId=self.client)[12], "enabled=",enableCollision)
        #             p.setCollisionFilterPair(self.quadruped, self.quadruped, l0,l1,enableCollision, physicsClientId=self.client)

        p.setRealTimeSimulation(0, physicsClientId=self.client) # this has no effect in DIRECT mode, only GUI mode
        # below is the default. Simulation params need to be retuned if this is changed
        p.setTimeStep(1/240., physicsClientId=self.client)

        for i in range (p.getNumJoints(self.quadruped, physicsClientId=self.client)):
            p.changeDynamics(self.quadruped,i,linearDamping=0, angularDamping=.5, physicsClientId=self.client)

        self.motor_joint_indices = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16] # the other joints in the urdf are fixed joints 
        self.n_motors = 12
        self.state_space_dim = 12 * 3 + 4 + 4 # (44) applied torque, pos, and vel for each motor, base orientation (quaternions), foot normal forces
        self.action_space_dim = self.n_motors
        self.actions_ub = np.empty(self.action_space_dim)
        self.actions_lb = np.empty(self.action_space_dim)
        self.action_mean = np.empty(self.action_space_dim)
        self.action_range = np.empty(self.action_space_dim)
        self.observations_ub = np.empty(self.state_space_dim)
        self.observations_lb = np.empty(self.state_space_dim)

        self.state = np.zeros(self.state_space_dim)
        self.applied_torques = np.zeros(self.n_motors)
        self.joint_velocities = np.zeros(self.n_motors)
        self.joint_positions = np.zeros(self.n_motors)
        self.base_orientation = np.zeros(4)
        self.foot_normal_forces = np.zeros(4)
        self.base_position = np.zeros(3) # not returned as observation, but used for other calculations
        self.previous_base_twist = np.zeros(6) # not returned as observation, but used for other calculations
        # self.previous_lower_limb_vels = np.zeros(4 * 6)
        # self.state_noise_std = 0.03125  * np.array([3.14, 40] * 12 + [0.78 * 0.25] * 4 + [0.25] * 3)
        self.perturbation_rate = 0.01 # probability that a random perturbation is applied to the torso
        self.max_torque = 40.0
        self.kp = 1.0 
        self.kd = 1.0

        self._find_space_limits()
        # self.num_envs = 1

        self.reward = 0 # this is to store most recent reward
        self.action_space = spaces.Box(
            low=-np.ones(self.action_space_dim),
            high=np.ones(self.action_space_dim),
            dtype=np.float32
            )

        self.observation_space = spaces.Box(
            low=self.observations_lb,
            high=self.observations_ub,
            dtype=np.float32
            )

    def _get_foot_contacts(self):
        '''Returns a numpy array of shape (4,) containing the normal forces on each foot with the ground. '''

        contacts = [0] * 4
        for i in range(len(self.foot_links)):
            info = p.getContactPoints(self.quadruped, 
                                        self.plane, 
                                        linkIndexA=self.foot_links[i],
                                        physicsClientId=self.client)
            if len(info) == 0: # leg does not contact ground
                contacts[i] = 0 
            elif len(info) == 1: # leg has one contact with ground
                contacts[i] = info[0][9] # contact normal force
            else: # use the contact point with the max normal force when there is more than one contact on a leg
                normals = [info[i][9] for i in range(len(info))]
                contacts[i] = max(normals)
                print('Number of contacts on one foot: %d' %len(info))
                print('Normal Forces: ', normals,'\n')
        contacts = np.array(contacts)
        if (contacts > 10_000).any():
            warnings.warn("Foot contact force of %.2f over 10,000 (maximum of observation space)" %max(contacts))

        # begin code for debugging foot contacts
        debug = [0] * 4
        for i in range(len(self.foot_links)):
            info = p.getContactPoints(self.quadruped, 
                                        self.plane, 
                                        linkIndexA=self.foot_links[i],
                                        physicsClientId=self.client)
            if len(info) == 0: # leg does not contact ground
                debug[i] = 0 
            elif len(info) == 1: # leg has one contact with ground
                debug[i] = info[0][9] # contact normal force
            else: # use the contact point with the max normal force when there is more than one contact on a leg
                normals = [info[i][9] for i in range(len(info))]
                debug[i] = normals
                # print('\nmultiple contacts' + '*' * 100 + '\n')
        debug = np.array(debug)
        return contacts, debug


    def step(self, action):
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        if not ((-1.0 <= action) & (action <= 1.0)).all():
            print("Action passed to env.step(): ", action)
            raise ValueError('Action is out-of-bounds of +/- 1.0') 
            
        p.setJointMotorControlArray(self.quadruped,
            self.motor_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=self._actions_to_positions(action),
            forces=self.max_torque * np.ones(self.n_motors),
            positionGains=self.kp * np.ones(12),
            velocityGains=self.kd * np.ones(12),
            physicsClientId=self.client)

        if (np.random.rand() > self.perturbation_rate) and self._apply_perturbations: 
            self._apply_perturbation()
        p.stepSimulation(physicsClientId=self.client)
        self._update_state()
        self.reward = self._reward_function()

        done, reason = self._is_state_terminal()
        info = {'':''} # this is returned so that env.step() matches Open AI gym API
        if done:
            info['termination_reason'] = reason
        return self.state, self.reward, done, info

    
    def _apply_perturbation(self):
        if np.random.rand() > 0.5: # apply force
            force = tuple(10 * (np.random.rand(3) - 0.5))
            p.applyExternalForce(self.quadruped, -1, force, (0,0,0), p.LINK_FRAME, physicsClientId=self.client)
        else: # apply torque
            torque = tuple(0.5 * (np.random.rand(3) - 0.5))
            p.applyExternalTorque(self.quadruped, -1, torque, p.LINK_FRAME, physicsClientId=self.client)


    def reset(self):
        '''Resets the robot to a neutral standing position, knees slightly bent. The motor control command is to 
        prevent the robot from jumping/falling on first user command. Simulation is stepped to allow robot to fall
        to ground and settle completely.'''

        starting_pos = [0.037199,    0.660252,   -1.200187,   -0.028954,    0.618814, 
          -1.183148,    0.048225,    0.690008,   -1.254787,   -0.050525,    0.661355,   -1.243304]
        p.resetBasePositionAndOrientation(self.quadruped,
                                            posObj=[0,0,0.48],
                                            ornObj=[0,0,0,1.0],
                                            physicsClientId=self.client) 
        for i in range(self.n_motors): # for some reason there is no p.resetJointStates (plural)
                        p.resetJointState(self.quadruped, 
                        self.motor_joint_indices[i],
                        starting_pos[i],
                        targetVelocity=0,
                        physicsClientId=self.client)
        p.setJointMotorControlArray(self.quadruped,
                                    self.motor_joint_indices,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=starting_pos,
                                    forces=self.max_torque * np.ones(self.n_motors),
                                    positionGains=self.kp * np.ones(12),
                                    velocityGains=self.kd * np.ones(12),
                                    physicsClientId=self.client)
        for i in range(500):
            p.stepSimulation(physicsClientId=self.client)
        self._update_state()
        return self.state


    def render(self, mode='human'):
        '''Setting the render kwarg in the constructor determines if the env will render or not.'''

        RENDER_WIDTH = 480 
        RENDER_HEIGHT = 360

        base_x_velocity = np.array(p.getBaseVelocity(self.quadruped, physicsClientId=self.client)).flatten()[0]
        torque_pen = -0.00001 * np.power(self.applied_torques, 2).mean()

        # RENDER_WIDTH = 960 
        # RENDER_HEIGHT = 720

        # RENDER_WIDTH = 1920
        # RENDER_HEIGHT = 1080

        if mode == 'rgb_array':
            base_pos, _ = p.getBasePositionAndOrientation(self.quadruped, physicsClientId=self.client)
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
            rgb_array = np.array(px)
            rgb_array = rgb_array[:, :, :3]
            img = putText(np.float32(rgb_array), 'X-velocity:' + str(base_x_velocity)[:6], (1, 60), 
                            FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
            img = putText(np.float32(img), 'Torque Penalty Term: ' + str(torque_pen)[:8], (1, 80), 
                            FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
            img = putText(np.float32(img), 'Total Rew: ' + str(torque_pen + base_x_velocity)[:8], (1, 100), 
                            FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
            _, foot_contacts = self._get_foot_contacts()
            for i in range(4):
                img = putText(np.float32(img), 
                            ('Foot %d contacts: ' %(i+1)) + str(foot_contacts[i].round(2)), 
                            (200, 60 + 20 * i), 
                            FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
            return np.uint8(img)

        else: 
            return


    def close(self):
        pass

    def _positions_to_actions(self, positions):
        return (positions - self.action_mean) / (self.action_range * 0.5)
  

    def _actions_to_positions(self, actions):
        return actions * (self.action_range * 0.5) + self.action_mean


    def _find_space_limits(self):
        ''' find upper and lower bounds of action and observation spaces''' 

       # find bounds of action space 
        for i in range(self.n_motors): 
            joint_info = p.getJointInfo(self.quadruped, self.motor_joint_indices[i], physicsClientId=self.client)
            # bounds on joint position
            self.actions_lb[i] = joint_info[8]
            self.actions_ub[i] = joint_info[9]
            
        # no joint limits given for the thigh joints, so set them to plus/minus 90 degrees
        for i in range(self.action_space_dim):
            if self.actions_ub[i] <= self.actions_lb[i]:
                self.actions_lb[i] = -3.14159 * 0.5
                self.actions_ub[i] = 3.14159 * 0.5

        self.action_mean = (self.actions_ub + self.actions_lb)/2 
        self.action_range = self.actions_ub - self.actions_lb

        # find the bounds of the state space (joint torque, joint position, joint velocity, base orientation)
        self.observations_lb = np.concatenate((-self.max_torque * np.ones(12), 
                                                self.actions_lb,
                                                -40 * np.ones(12), 
                                                -0.78 * np.ones(4),
                                                np.zeros(4)))

        self.observations_ub = np.concatenate((self.max_torque * np.ones(12), 
                                                self.actions_ub, 
                                                40 * np.ones(12), 
                                                0.78 * np.ones(4),
                                                1e4 * np.ones(4)))


    def _reward_function(self) -> float:
        ''' Calculates reward based off of current state '''
        base_twist = np.array(p.getBaseVelocity(self.quadruped, physicsClientId=self.client)).flatten()
        base_x_velocity = base_twist[0]
        # base_y_velocity = base_twist[1]
        # base_accel_penalty = np.power(base_twist[1:] - self.previous_base_twist[1:], 2).mean()
        torque_penalty = np.power(self.applied_torques, 2).mean()
        # lower_limb_states = list(p.getLinkStates(self.quadruped, self.lower_legs, computeLinkVelocity=True))
        # lower_limb_vels = np.array([lower_limb_states[i][6] + lower_limb_states[i][7] for i in range(4)]).flatten()
        # lower_limb_accel_penalty = np.power(lower_limb_vels - self.previous_lower_limb_vels, 2).mean()
        # orientation =  np.array(list(p.getEulerFromQuaternion(self.base_orientation)))

        # lower_limb_height_bonus = np.array([lower_limb_states[i][0][2] for i in range(4)]).mean()
        # orientation_pen = np.sum(np.power(orientation, 2))
        # self.previous_base_twist = base_twist 
        # self.previous_lower_limb_vels = lower_limb_vels
        # print(base_x_velocity , 0.0001 * torque_penalty , 0.01 * base_accel_penalty , 0.01 * lower_limb_accel_penalty, 0.1 * lower_limb_height_bonus)
        existence_reward = 1.0
        return base_x_velocity - 0.00001 * torque_penalty + existence_reward #-0.01*orientation_pen#- 0.01 * base_accel_penalty \
             # - 0.01 * lower_limb_accel_penalty - 0.1 * abs(base_y_velocity) # \
             # + 0.1 * lower_limb_height_bonus


    def _update_state(self):

        joint_states = p.getJointStates(self.quadruped, self.motor_joint_indices, physicsClientId=self.client)
        self.applied_torques  = np.array([joint_states[i][3] for i in range(self.n_motors)])
        self.joint_positions  = np.array([joint_states[i][0] for i in range(self.n_motors)])
        self.joint_velocities = np.array([joint_states[i][1] for i in range(self.n_motors)])

        base_position, base_orientation = p.getBasePositionAndOrientation(self.quadruped, physicsClientId=self.client)
        self.base_position = np.array(base_position)
        self.base_orientation = np.array(base_orientation)

        self.foot_normal_forces, _ = self._get_foot_contacts()

        self.state = np.concatenate((self.applied_torques, 
                                    self.joint_positions,
                                    self.joint_velocities,
                                    self.base_orientation,
                                    self.foot_normal_forces))
        if np.isnan(self.state).any():
            print('nans in state')
            breakpoint()


    def _is_state_terminal(self) -> bool:
        ''' Calculates whether to end current episode due to failure based on current state. Does not consider timeout.
        Returns boolean and reason if True '''

        base_z_position = self.base_position[2]
        height_out_of_bounds = (base_z_position < 0.23) or (base_z_position > 0.8)
        # 0.78 rad is about 45 deg
        # falling = (abs(np.array(p.getEulerFromQuaternion(self.base_orientation))) > [0.78*2, 0.78, 0.78]).any() 
        falling = (abs(np.array(p.getEulerFromQuaternion(self.base_orientation))) > 0.78).any() 
        if falling:
            reason = 'falling'
            # print(reason)
            # print(p.getEulerFromQuaternion(self.base_orientation))
        elif height_out_of_bounds:
            reason = 'height_out_of_bounds'
        else:
            reason = ''
        return (falling or height_out_of_bounds), reason


if __name__ == '__main__':
    '''Perform check by feeding in the mocap trajectory provided by Unitree (linked) into the aliengo robot and
    save video. https://github.com/unitreerobotics/aliengo_pybullet'''

    import cv2
    env = gym.make('gym_aliengo:aliengo-v0')
    env.reset()
    img = env.render('rgb_array')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_list = [img]
    counter = 0

    # for i in range(200):
    #     p.stepSimulation(physicsClientId=env.client)
    #     img = env.render('rgb_array')
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     img_list.append(img)

    # with open('mocap.txt','r') as f:
    #     env.step(env._positions_to_actions(np.array(line.split(',')[2:],dtype=np.float32)))
    # img = env.render('rgb_array')
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # img_list.append(img)
    # for i in range(200):
    #     p.stepSimulation(physicsClientId=env.client)
    #     img = env.render('rgb_array')
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     img_list.append(img)
        
    with open('mocap.txt','r') as f:
        for line in f:
            positions =  env._positions_to_actions(np.array(line.split(',')[2:], dtype=np.float32))
            obs,_ , done, _ =  env.step(positions)
            if counter%4 ==0: # sim runs 240 Hz, want 60 Hz vid   
                img = env.render('rgb_array')
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_list.append(img)
            counter +=1
            # if counter ==100:
            #     break

    height, width, layers = img.shape
    size = (width, height)
    out = cv2.VideoWriter('test_vid.avi', cv2.VideoWriter_fourcc(*'XVID'), 60, size)

    for img in img_list:
        out.write(img)
    out.release()
    print('Video saved')


    
    



    


