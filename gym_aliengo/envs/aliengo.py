'''
Implements the class for the Aliengo robot to be used in all the environments in this repo. All inputs and outputs 
should be numpy arrays.
'''

import pybullet as p
import numpy as np
import os

class Aliengo:
    def __init__(self, pybullet_client, max_torque=40.0, kd=1.0, kp=1.0):
        self.kp = kp
        self.kd = kd
        self.client = pybullet_client
        self.max_torque = max_torque
        self.n_motors = 12

        self.foot_links = [5, 9, 13, 17]
        self.quadruped = self.load_urdf()

    
        
        # indices are in order of [shoulder, hip, knee] for FR, FL, RR, RL. The skipped numbers are fixed joints
        # in the URDF
        self.motor_joint_indices = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16] 
       

        self.positions_lb, self.positions_ub, self.position_mean, self.position_range = self._find_position_bounds()


    def render(self, mode, client): 
        '''Returns RGB array of current scene if mode is 'rgb_array'.'''

        RENDER_WIDTH = 480 
        RENDER_HEIGHT = 360

        # base_x_velocity = np.array(self.client.getBaseVelocity(self.quadruped)).flatten()[0]
        # torque_pen = -0.00001 * np.power(self.applied_torques, 2).mean()

        # RENDER_WIDTH = 960 
        # RENDER_HEIGHT = 720

        # RENDER_WIDTH = 1920
        # RENDER_HEIGHT = 1080

        if mode == 'rgb_array':
            base_pos, _ = self.client.getBasePositionAndOrientation(self.quadruped)
            # base_pos = self.minitaur.GetBasePosition()
            view_matrix = client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_pos,
                distance=2.0,
                yaw=0,
                pitch=-30.,
                roll=0,
                upAxisIndex=2)
            proj_matrix = client.computeProjectionMatrixFOV(fov=60,
                aspect=float(RENDER_WIDTH) /
                RENDER_HEIGHT,
                nearVal=0.1,
                farVal=100.0)
            _, _, px, _, _ = client.getCameraImage(width=RENDER_WIDTH,
                height=RENDER_HEIGHT,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL)
            img = np.array(px)
            img = img[:, :, :3]
            # img = putText(np.float32(img), 'X-velocity:' + str(base_x_velocity)[:6], (1, 60), 
            #                 FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
            # img = putText(np.float32(img), 'Torque Penalty Term: ' + str(torque_pen)[:8], (1, 80), 
            #                 FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
            # img = putText(np.float32(img), 'Total Rew: ' + str(torque_pen + base_x_velocity)[:8], (1, 100), 
            #                 FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
            # foot_contacts = self._get_foot_contacts()
            # for i in range(4):
            #     if type(foot_contacts[i]) is list: # multiple contacts
            #         assert False
            #         num = np.array(foot_contacts[i]).round(2)
            #     else:
            #         num = round(foot_contacts[i], 2)
            #     img = putText(np.float32(img), 
            #                 ('Foot %d contacts: ' %(i+1)) + str(num), 
            #                 (200, 60 + 20 * i), 
            #                 FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
            # img = putText(np.float32(img), 
            #                 'Body Contact: ' + str(self._is_non_foot_ground_contact()), 
            #                 (200, 60 + 20 * 4), 
            #                 FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
            # img = putText(np.float32(img), 
            #                 'Self Collision: ' + str(self.quadruped.self_collision()), 
            #                 (200, 60 + 20 * 5), 
            #                 FONT_HERSHEY_SIMPLEX, 0.375, (0,0,0))
            return np.uint8(img)

        else: 
            return

    def load_urdf(self):
        urdfFlags = p.URDF_USE_SELF_COLLISION
        quadruped= self.client.loadURDF(os.path.join(os.path.dirname(__file__), '../urdf/aliengo.urdf'),
                                    basePosition=[0,0, 0.48], 
                                    baseOrientation=[0,0,0,1], 
                                    flags = urdfFlags, 
                                    useFixedBase=False)

        self.foot_links = [5, 9, 13, 17]

        for i in range (self.client.getNumJoints(quadruped)):
            self.client.changeDynamics(quadruped, i, linearDamping=0, angularDamping=.5)

        
        # self.quadruped.foot_links = [5, 9, 13, 17]
        # self.lower_legs = [2,5,8,11]
        # for l0 in self.lower_legs:
        #     for l1 in self.lower_legs:
        #         if (l1>l0):
        #             enableCollision = 1
        #             # print("collision for pair",l0,l1, self.client.getJointInfo(self.quadruped,l0, physicsClientId=self.client)[12],p.getJointInfo(self.quadruped,l1, physicsClientId=self.client)[12], "enabled=",enableCollision)
        #             self.client.setCollisionFilterPair(self.quadruped, self.quadruped, l0,l1,enableCollision, physicsClientId=self.client)
        
        return quadruped

    def remove_body(self):
        self.client.removeBody(self.quadruped)
        



    def self_collision(self):
        '''Returns true if any of the robot links are colliding with any other link'''

        points = self.client.getContactPoints(self.quadruped, self.quadruped)
        return len(points) > 0

    def set_joint_position_targets(self, positions):
        '''
        Takes positions in range of [-1, 1]. These positions are mapped to the actual range of joint positions for 
        each joint of the robot. 
        '''

        assert isinstance(positions, np.ndarray)
        assert ((-1.0 <= positions) & (positions <= 1.0)).all(), '\nposition received: ' + str(positions) + '\n'
 
        positions = self._actions_to_positions(positions)

        self.client.setJointMotorControlArray(self.quadruped,
            self.motor_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=positions,
            forces=self.max_torque * np.ones(self.n_motors),
            positionGains=self.kp * np.ones(self.n_motors),
            velocityGains=self.kd * np.ones(self.n_motors))


    def _positions_to_actions(self, positions):
        '''Maps actual robot joint positions in radians to the range of [-1.0, 1.0]'''

        return (positions - self.position_mean) / (self.position_range * 0.5)


    def _actions_to_positions(self, actions):
        '''
        Takes actions or normalized positions in the range of [-1.0, 1.0] and maps them to actual joint positions in 
        radians.
        '''

        return actions * (self.position_range * 0.5) + self.position_mean

    
    def _find_position_bounds(self):
        positions_lb = np.zeros(self.n_motors)
        positions_ub = np.zeros(self.n_motors)
        # find bounds of action space 
        for i in range(self.n_motors): 
            joint_info = self.client.getJointInfo(self.quadruped, self.motor_joint_indices[i])
            # bounds on joint position
            positions_lb[i] = joint_info[8]
            positions_ub[i] = joint_info[9]
            
        # no joint limits given for the thigh joints, so set them to plus/minus 90 degrees
        for i in range(self.n_motors):
            if positions_ub[i] <= positions_lb[i]:
                positions_lb[i] = -3.14159 * 0.5
                positions_ub[i] = 3.14159 * 0.5

        position_mean = (positions_ub + positions_lb)/2 
        position_range = positions_ub - positions_lb

        return positions_lb, positions_ub, position_mean, position_range


    def get_joint_position_bounds(self):
        '''Returns lower and upper bounds of the allowable joint positions as a numpy array. '''

        return -np.ones(self.n_motors), np.ones(self.n_motors)


    def get_joint_velocity_bounds(self):
        '''The value 40 is from the Aliengo URDF.'''

        return -np.ones(self.n_motors) * 40, np.ones(self.n_motors) * 40
    

    def get_joint_torque_bounds(self):
        '''
        Returns lower and upper bounds of the allowable joint torque as a numpy array.
        Note: I am not sure if I need to distinguish between the bounds of the torque you are allowed to set vs the 
        bounds of the applied torque that can occur in simulation. In other words, does pybullet allow the applied 
        torque to sometimes go slightly out of the bounds of self.max_torque? 
         '''

        return - np.ones(self.n_motors) * self.max_torque, np.ones(self.n_motors) * self.max_torque

    
    def get_joint_states(self):
        '''Note: Reaction forces will return all zeros unless a torque sensor has been set'''

        joint_states = self.client.getJointStates(self.quadruped, self.motor_joint_indices)
        joint_positions  = self._positions_to_actions(np.array([joint_states[i][0] for i in range(self.n_motors)]))
        joint_velocities = np.array([joint_states[i][1] for i in range(self.n_motors)])
        reaction_forces  = np.array([joint_states[i][2] for i in range(self.n_motors)])
        applied_torques  = np.array([joint_states[i][3] for i in range(self.n_motors)])
        return joint_positions, joint_velocities, reaction_forces, applied_torques


    def get_base_position_and_orientation(self):
        base_position, base_orientation = self.client.getBasePositionAndOrientation(self.quadruped)    
        return np.array(base_position), np.array(base_orientation)
    
    
    def get_base_twist(self):
        return np.array(self.client.getBaseVelocity(self.quadruped)).flatten()

        
    def reset_joint_positions(self, positions=None, stochastic=True):
        '''Note: This ignores any physics or controllers and just overwrites joint positions to the given value''' 

        if positions: 
            positions = self._actions_to_positions(positions)
        else: 
            # use the default starting position, knees slightly bent, from first line of mocap file
            positions = np.array([0.037199,    0.660252,   -1.200187,   -0.028954,    0.618814, 
                            -1.183148,    0.048225,    0.690008,   -1.254787,   -0.050525,    0.661355,   -1.243304])

        if stochastic: 
            # add random noise to positions
            noise = (np.random.rand(12) - 0.5) * self.position_range * 0.05
            positions += noise
            positions = np.clip(positions, self.positions_lb, self.positions_ub)



        for i in range(self.n_motors): # for some reason there is no p.resetJointStates (plural)
            self.client.resetJointState(self.quadruped, 
                                self.motor_joint_indices[i],
                                positions[i],
                                targetVelocity=0)

        ''' TODO: see if the following is actually necessary. i.e. does pybullet retain motor control targets after you 
         Reset joint positions? If so, the following is necessary'''

        self.client.setJointMotorControlArray(self.quadruped,
                                    self.motor_joint_indices,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=positions,
                                    forces=self.max_torque * np.ones(self.n_motors),
                                    positionGains=self.kp * np.ones(12),
                                    velocityGains=self.kd * np.ones(12))






    