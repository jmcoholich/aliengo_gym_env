'''
Implements the class for the Aliengo robot to be used in all the environments in this repo. All inputs and outputs 
should be numpy arrays.
'''

import pybullet as p
import numpy as np
import os
import time

class Aliengo:
    def __init__(self, pybullet_client, max_torque=40.0, kd=1.0, kp=1.0, 
                    fixed=False, fixed_position=[0,0,1.0], fixed_orientation=[0,0,0]):
        self.kp = kp
        self.kd = kd
        self.client = pybullet_client
        self.max_torque = max_torque
        self.n_motors = 12

        self.foot_links = [5, 9, 13, 17]
        self.hip_links =  [3, 7, 11, 15]
        self.quadruped = self.load_urdf(fixed=fixed, fixed_position=fixed_position, fixed_orientation=fixed_orientation)

    
        
        # indices are in order of [shoulder, hip, knee] for FR, FL, RR, RL. The skipped numbers are fixed joints
        # in the URDF
        self.motor_joint_indices = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16] 
        self.hip_joints = [2, 6, 10, 14]
        self.thigh_joints = [3, 7, 11, 15]
        self.knee_joints = [4, 8, 12, 16]
        self.num_links = 19
        self.positions_lb, self.positions_ub, self.position_mean, self.position_range = self._find_position_bounds()


        self._debug_ids = [] # this is for the visualization when debug = True for heightmap

        self.first_debug = True # this is for debug for set foot positions

    
    def set_foot_positions(self, foot_positions):
        '''Takes a numpy array of shape (4, 3) which represents foot xyz relative to the hip joint. Uses IK to 
        calculate joint position targets and sets those targets. Does not return anything'''


        assert foot_positions.shape == (4,3)
        debug=True # render/display things
        hip_joint_positions = np.zeros((4, 3)) # storing these for use when debug
        commanded_global_foot_positions = np.zeros((4, 3))
        for i in range(4):
            hip_offset_from_base = self.client.getJointInfo(self.quadruped, self.hip_joints[i])[14]
            base_p, base_o = self.client.getBasePositionAndOrientation(self.quadruped)
            hip_joint_positions[i], _ = np.array(self.client.multiplyTransforms(positionA=base_p,
                                                    orientationA=base_o,
                                                    positionB=hip_offset_from_base,
                                                    orientationB=[0.0, 0.0, 0.0, 1.0]))
            # rotate the input foot_positions x and y from robot yaw direction to global coordinate frame 
            _, _, yaw = self.client.getEulerFromQuaternion(base_o)
            commanded_global_foot_positions[i][0] = hip_joint_positions[i][0] + \
                                                foot_positions[i][0] * np.cos(yaw) + foot_positions[i][1] * np.sin(yaw)
            commanded_global_foot_positions[i][1] = hip_joint_positions[i][1] + \
                                                foot_positions[i][0] * np.sin(yaw) + foot_positions[i][1] * np.cos(yaw)
            commanded_global_foot_positions[i][2] = hip_joint_positions[i][2] + foot_positions[i][2] + 0.0265 # collision radius
        # print(yaw)
        # print(commanded_global_foot_positions)
        # sys.exit()
        joint_positions = np.array(self.client.calculateInverseKinematics2(self.quadruped,
                                                self.foot_links,
                                                commanded_global_foot_positions))
                                                # maxNumIterations=1000,
                                                # residualThreshold=1e-10))
        self.set_joint_position_targets(joint_positions, true_positions=True)

        if debug:
            # green spheres are commanded positions, red spheres are actual positions
            if self.first_debug:
                commanded_ball = self.client.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 100, 0, 1.0])
                actual_ball = self.client.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[255, 0, 0, 1.0])
                for i in range(self.num_links):
                    self.client.changeVisualShape(self.quadruped, i, rgbaColor=[0, 0, 0, 0.75])
                # visualize commanded foot positions 
                self.foot_ball_ids = [0]*4
                self.hip_ball_ids = [0]*4
                for i in range(4):
                    self.foot_ball_ids[i] = self.client.createMultiBody(baseVisualShapeIndex=commanded_ball, 
                                                                    basePosition=commanded_global_foot_positions[i])
                # visualize calculated hip positions
                for i in range(4):
                    self.hip_ball_ids[i] = self.client.createMultiBody(baseVisualShapeIndex=actual_ball, 
                                                                        basePosition=hip_joint_positions[i])
                self.first_debug = False
            else:
                for i in range(4):
                    self.client.resetBasePositionAndOrientation(self.foot_ball_ids[i], 
                                                                posObj=commanded_global_foot_positions[i], 
                                                                ornObj=[0,0,0,1])
                    self.client.resetBasePositionAndOrientation(self.hip_ball_ids[i], 
                                                                posObj=hip_joint_positions[i], 
                                                                ornObj=[0,0,0,1])
            
            



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


    def _apply_perturbation(self):
        raise NotImplementedError
        if np.random.rand() > 0.5: # apply force
            force = tuple(10 * (np.random.rand(3) - 0.5))
            self.client.applyExternalForce(self.quadruped, -1, force, (0,0,0), p.LINK_FRAME)
        else: # apply torque
            torque = tuple(0.5 * (np.random.rand(3) - 0.5))
            self.client.applyExternalTorque(self.quadruped, -1, torque, p.LINK_FRAME)
    

    def _get_foot_contacts(self, _object=None): 
        '''
        Returns a numpy array of shape (4,) containing the normal forces on each foot with the object given. If 
        no object given, just checks with any object in self.client simulation. 
        '''

        contacts = [0] * 4
        for i in range(len(self.foot_links)):
            if _object is None:
                info = self.client.getContactPoints(bodyA=self.quadruped, 
                                                    linkIndexA=self.foot_links[i])
            else:
                info = self.client.getContactPoints(bodyA=self.quadruped, 
                                                    bodyB=_object,
                                                    linkIndexA=self.foot_links[i])
            if len(info) == 0: # leg does not contact ground
                contacts[i] = 0 
            elif len(info) == 1: # leg has one contact with ground
                contacts[i] = info[0][9] # contact normal force
            else: # use the contact point with the max normal force when there is more than one contact on a leg 
                #TODO investigate scenarios with more than one contact point and maybe do something better (mean 
                # or norm of contact forces?)
                normals = [info[i][9] for i in range(len(info))] 
                contacts[i] = max(normals)
                # print('Number of contacts on one foot: %d' %len(info))
                # print('Normal Forces: ', normals,'\n')
        contacts = np.array(contacts)
        if (contacts > 10_000).any():
            warnings.warn("Foot contact force of %.2f over 10,000 (maximum of observation space)" %max(contacts))
        return contacts 


    def _get_heightmap(self, client, ray_start_height, base_position, heightmap_params):
        '''Debug flag enables printing of labeled coordinates and measured heights to rendered simulation. 
        Uses the "fake_client" simulation instance in order to avoid measuring the robot instead of terrain
        ray_start_height should be a value that is guranteed to be above any terrain we want to measure. 
        It is also where the debug text will be displayed when debug=True.'''

        length = heightmap_params['length']
        robot_position = heightmap_params['robot_position']
        grid_spacing = heightmap_params['grid_spacing']
        assert length % grid_spacing == 0
        grid_len = int(length/grid_spacing) + 1

        debug = False
        show_xy = False

        if self._debug_ids != []: # remove the exiting debug items
            for _id in self._debug_ids:
                self.client.removeUserDebugItem(_id)
            self._debug_ids = []

        base_x = base_position[0] 
        base_y = base_position[1]
        base_z = base_position[2]


        x = np.linspace(0, length, grid_len)
        y = np.linspace(-length/2.0, length/2.0, grid_len)
        coordinates = np.array(np.meshgrid(x,y))
        coordinates[0,:,:] += base_x - robot_position
        coordinates[1,:,:] += base_y  
        # coordinates has shape (2, grid_len, grid_len)
        coor_list = coordinates.reshape((2, grid_len**2)).swapaxes(0, 1) # is now shape (grid_len**2,2) 
        ray_start = np.append(coor_list, np.ones((grid_len**2, 1)) * ray_start_height, axis=1) #TODO check that this and in general the values are working properly
        ray_end = np.append(coor_list, np.zeros((grid_len**2, 1)) - 1, axis=1)
        raw_output = client.rayTestBatch(ray_start, ray_end) 
        z_heights = np.array([raw_output[i][3][2] for i in range(grid_len**2)])
        relative_z_heights = z_heights - base_z

        if debug:
            # #print xy coordinates of robot origin 
            # _id = self.client.addUserDebugText(text='%.2f, %.2f'%(base_x, base_y),
            #             textPosition=[base_x, base_y,ray_start_height+1],
            #             textColorRGB=[0,0,0])
            # self._debug_ids.append(_id)
            for i in range(grid_len):
                for j in range(grid_len):
                    if show_xy:
                        text = '%.3f, %.3f, %.3f'%(coordinates[0,i,j], coordinates[1,i,j], z_heights.reshape((grid_len, grid_len))[i,j])
                    else:
                        text = '%.3f'%(z_heights.reshape((grid_len, grid_len))[i,j])
                    _id = self.client.addUserDebugText(text=text,
                                            textPosition=[coordinates[0,i,j], coordinates[1,i,j],ray_start_height+0.5],
                                            textColorRGB=[0,0,0])
                    self._debug_ids.append(_id)
                    _id = self.client.addUserDebugLine([coordinates[0,i,j], coordinates[1,i,j],ray_start_height+0.5],
                                            [coordinates[0,i,j], coordinates[1,i,j], 0],
                                            lineColorRGB=[0,0,0] )
                    self._debug_ids.append(_id)

        return relative_z_heights.reshape((grid_len, grid_len))


    def _is_non_foot_ground_contact(self): #TODO if I ever use this in this env, account for stepping stone contact
        """Detect if any parts of the robot, other than the feet, are touching the ground."""

        raise NotImplementedError
        contact = False
        for i in range(self.num_joints):
            if i in self.foot_links: # the feet themselves are allow the touch the ground
                continue
            points = self.client.getContactPoints(self.quadruped, self.plane, linkIndexA=i)
            if len(points) != 0:
                contact = True
        return contact


    def load_urdf(self, fixed=False, fixed_position=[0,0,1.0], fixed_orientation=[0,0,0]):
        urdfFlags = p.URDF_USE_SELF_COLLISION
        if fixed:
            quadruped= self.client.loadURDF(os.path.join(os.path.dirname(__file__), '../urdf/aliengo.urdf'),
                                        basePosition=fixed_position, 
                                        baseOrientation=self.client.getQuaternionFromEuler(fixed_orientation), 
                                        flags = urdfFlags, 
                                        useFixedBase=True)
        else:
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

    def set_joint_position_targets(self, positions, true_positions=False):
        '''
        Takes positions in range of [-1, 1]. These positions are mapped to the actual range of joint positions for 
        each joint of the robot. 
        '''

        assert isinstance(positions, np.ndarray)
        
        if not true_positions:
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


def sine_tracking_test(client, quadruped):
    # test foot position command tracking and print tracking error
    t = 0
    while True:
        command = np.array([[0.1 * np.sin(2*t), -0.1 * np.sin(2*t), -0.3  + 0.1 * np.sin(2*t)] for _ in range(4)])
        quadruped.set_foot_positions(command)
        orientation = client.getQuaternionFromEuler([np.pi/4.*np.sin(t)]*3)
        client.resetBasePositionAndOrientation(quadruped.quadruped,[-1,1,1], orientation)
        time.sleep(1/240.)
        t += 1/240.

        calculate_tracking_error(command, client, quadruped)


def floor_tracking_test(client, quadruped):
    # test foot position command tracking and print tracking error
    t = 0
    while True:
        z = -0.500 # decreasing this to -0.51 should show feet collision with ground and inability to track
        command = np.array([[0.1 * np.sin(2*t), 0, z] for _ in range(4)])
        quadruped.set_foot_positions(command)
        client.resetBasePositionAndOrientation(quadruped.quadruped,[0.,0.,0.5], [0.,0.,0.,1.0])
        time.sleep(1/240.)
        t += 1/240.

        calculate_tracking_error(command, client, quadruped)

        
def calculate_tracking_error(foot_positions, client, quadruped):
    # calculate tracking error. First calculate the command in global coordinates
        hip_joint_positions = np.zeros((4, 3)) # storing these for use when debug
        commanded_global_foot_positions = np.zeros((4, 3))
        for i in range(4):
            hip_offset_from_base = client.getJointInfo(quadruped.quadruped, quadruped.hip_joints[i])[14]
            base_p, base_o = client.getBasePositionAndOrientation(quadruped.quadruped)
            hip_joint_positions[i], _ = np.array(client.multiplyTransforms(positionA=base_p,
                                                    orientationA=base_o,
                                                    positionB=hip_offset_from_base,
                                                    orientationB=[0.0, 0.0, 0.0, 1.0]))
            # rotate the input foot_positions x and y from robot yaw direction to global coordinate frame 
            _, _, yaw = client.getEulerFromQuaternion(base_o)
            commanded_global_foot_positions[i][0] = hip_joint_positions[i][0] + \
                                                foot_positions[i][0] * np.cos(yaw) + foot_positions[i][1] * np.sin(yaw)
            commanded_global_foot_positions[i][1] = hip_joint_positions[i][1] + \
                                                foot_positions[i][0] * np.sin(yaw) + foot_positions[i][1] * np.cos(yaw)
            commanded_global_foot_positions[i][2] = hip_joint_positions[i][2] + foot_positions[i][2]
        actual_pos = np.array([i[0] for i in client.getLinkStates(quadruped.quadruped, quadruped.foot_links)])
        print('Maximum tracking error: {:e}'.format(np.abs((commanded_global_foot_positions- actual_pos)).max()))


if __name__ == '__main__':
    # set up the quadruped in a pybullet simulation
    from pybullet_utils import bullet_client as bc
    client = client = bc.BulletClient(connection_mode=p.GUI)
    client.setTimeStep(1/240.)
    client.setGravity(0,0,-9.8)
    client.setRealTimeSimulation(True) # this has no effect in DIRECT mode, only GUI mode
    plane = client.loadURDF(os.path.join(os.path.dirname(__file__), '../urdf/plane.urdf'))
    quadruped = Aliengo(client, fixed=True, fixed_orientation=[0] * 3, fixed_position=[-1,1,1])

    # sine_tracking_test(client, quadruped)
    floor_tracking_test(client, quadruped)

    

    