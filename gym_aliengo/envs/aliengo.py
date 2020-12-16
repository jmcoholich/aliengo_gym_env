'''
Implements the class for the Aliengo robot to be used in all the environments in this repo. All inputs and outputs 
should be numpy arrays.
'''
'''
dev notes
- The changed env should function the exact same as the original one. 
    - test this by running the same policy on it and make sure it works. 
    - also just run aliengo_env.py for the mocap test case. 
***There is one major difference: This new implementation will return joint_positions as mapped from [-1, 1] even 
for the observation***
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
        urdfFlags = p.URDF_USE_SELF_COLLISION
        self.quadruped = p.loadURDF(os.path.join(os.path.dirname(__file__), '../urdf/aliengo.urdf'),
                                    basePosition=[0,0,0.48], 
                                    baseOrientation=[0,0,0,1], 
                                    flags = urdfFlags, 
                                    useFixedBase=False,
                                    physicsClientId=self.client)

        self.foot_links = [5, 9, 13, 17]

        for i in range (p.getNumJoints(self.quadruped, physicsClientId=self.client)):
            p.changeDynamics(self.quadruped, i, linearDamping=0, angularDamping=.5, physicsClientId=self.client)

        self.motor_joint_indices = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16] 
        # ^the other joints listed in the urdf are fixed joints 

        self.positions_lb, self.positions_ub, self.position_mean, self.position_range = self._find_position_bounds()



    def self_collision(self):
        '''Returns true if any of the robot links are colliding with any other link'''

        points = p.getContactPoints(self.quadruped, self.quadruped, physicsClientId=self.client)
        return len(points) > 0

    def set_joint_position_targets(self, positions):
        '''
        Takes positions in range of [-1, 1]. These positions are mapped to the actual range of joint positions for 
        each joint of the robot. 
        '''

        assert isinstance(positions, np.ndarray)
        assert ((-1.0 <= positions) & (positions <= 1.0)).all(), '\nposition received: ' + str(positions) + '\n'
 
        positions = self._actions_to_positions(positions)

        p.setJointMotorControlArray(self.quadruped,
            self.motor_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=positions,
            forces=self.max_torque * np.ones(self.n_motors),
            positionGains=self.kp * np.ones(self.n_motors),
            velocityGains=self.kd * np.ones(self.n_motors),
            physicsClientId=self.client)


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
            joint_info = p.getJointInfo(self.quadruped, self.motor_joint_indices[i], physicsClientId=self.client)
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

        joint_states = p.getJointStates(self.quadruped, self.motor_joint_indices, physicsClientId=self.client)
        joint_positions  = self._positions_to_actions(np.array([joint_states[i][0] for i in range(self.n_motors)]))
        joint_velocities = np.array([joint_states[i][1] for i in range(self.n_motors)])
        reaction_forces  = np.array([joint_states[i][2] for i in range(self.n_motors)])
        applied_torques  = np.array([joint_states[i][3] for i in range(self.n_motors)])
        return joint_positions, joint_velocities, reaction_forces, applied_torques


    def get_base_position_and_orientation(self):
        base_position, base_orientation = p.getBasePositionAndOrientation(self.quadruped, physicsClientId=self.client)    
        return np.array(base_position), np.array(base_orientation)
    
    
    def get_base_twist(self):
        return np.array(p.getBaseVelocity(self.quadruped, physicsClientId=self.client)).flatten()

        
    def reset_joint_positions(self, positions=None):
        '''Note: This ignores any physics or controllers and just overwrites joint positions to the given value''' 

        if positions: 
            positions = self._actions_to_positions(positions)
        else: 
            # use the default starting position, knees slightly bent, from first line of mocap file
            positions = [0.037199,    0.660252,   -1.200187,   -0.028954,    0.618814, 
                            -1.183148,    0.048225,    0.690008,   -1.254787,   -0.050525,    0.661355,   -1.243304]

        for i in range(self.n_motors): # for some reason there is no p.resetJointStates (plural)
            p.resetJointState(self.quadruped, 
                                self.motor_joint_indices[i],
                                positions[i],
                                targetVelocity=0,
                                physicsClientId=self.client)

        ''' TODO: see if the following is actually necessary. i.e. does pybullet retain motor control targets after you 
         Reset joint positions? If so, the following is necessary'''

        p.setJointMotorControlArray(self.quadruped,
                                    self.motor_joint_indices,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=positions,
                                    forces=self.max_torque * np.ones(self.n_motors),
                                    positionGains=self.kp * np.ones(12),
                                    velocityGains=self.kd * np.ones(12),
                                    physicsClientId=self.client)






    