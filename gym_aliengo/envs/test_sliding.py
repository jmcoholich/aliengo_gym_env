'''Script created to test and debug issue with aliengo feet sliding with action repeat. '''

from pybullet_utils import bullet_client as bc
import pybullet as p
import numpy as np
import os
import time
from aliengo import Aliengo
import sys

def update_collision_sphere_vis(client, quadruped, sphere_ids):
    # get foot positions
    actual_pos = np.array([i[0] for i in client.getLinkStates(quadruped.quadruped, quadruped.foot_links)])
    for i in range(4):
        client.resetBasePositionAndOrientation(sphere_ids[i], posObj=actual_pos[i], ornObj=[0,0,0,1])


def change_dynamics(client, quadruped, plane):
    # print default dyanmics
    
    info = client.getDynamicsInfo(quadruped.quadruped, quadruped.foot_links[0])
    print('\n' + '#'*100, '\nDefault Foot Dynamics Values: ')
    print('lateral friction: {}'.format(info[1]))
    print('rolling friction: {}'.format(info[6]))
    print('spinning friction: {}\n'.format(info[7]) + '#'*100)

    info = client.getDynamicsInfo(plane, -1)
    print('Default Plane Dynamics Values: ')
    print('lateral friction: {}'.format(info[1]))
    print('rolling friction: {}'.format(info[6]))
    print('spinning friction: {}\n'.format(info[7]) + '#'*100)

    lateral_friction = 10000.0
    rolling_friction = 0.01 * 1e5
    spinning_friction = 0.01 * 1e5
    for link in quadruped.foot_links: 
        client.changeDynamics(quadruped.quadruped, 
                                link, 
                                lateralFriction=lateral_friction, 
                                rollingFriction=rolling_friction,
                                spinningFriction=spinning_friction)
    # client.changeDynamics(plane, 
    #                         -1, 
    #                         # lateralFriction=lateral_friction, 
    #                         rollingFriction=rolling_friction,
    #                         spinningFriction=spinning_friction)
    print('New Values:')
    # print('lateral friction: {}'.format(lateral_friction))
    print('rolling friction: {}'.format(rolling_friction))
    print('spinning friction: {}\n'.format(spinning_friction) + '#'*100)    


def fix_feet_points(client, quadruped):
    actual_pos = np.array([i[0] for i in client.getLinkStates(quadruped.quadruped, quadruped.foot_links)])
    for i in range(4):
        cstr = client.createConstraint(parentBodyUniqueId=quadruped.quadruped,
                                parentLinkIndex=quadruped.foot_links[i],
                                childBodyUniqueId=-1,
                                childLinkIndex=-1,
                                jointType=p.JOINT_POINT2POINT,
                                jointAxis=[0,0,1], # not sure if this does anything for point2point or fixed
                                parentFramePosition=[0,0, -0.0265], # point on foot to constrain
                                childFramePosition=actual_pos[i] + np.array([0,0, -0.0265]))
        # print('erp is ', client.getConstraintInfo(cstr)[-1])
        # print('max force ', client.getConstraintInfo(cstr)[-5])
        client.changeConstraint(cstr, erp=1e5, maxForce=500*1000)
        # break


client = bc.BulletClient(connection_mode=p.GUI)
client.setTimeStep(1/240.)
client.setGravity(0,0,-9.8)
client.setRealTimeSimulation(False) # this has no effect in DIRECT mode, only GUI mode
plane = client.loadURDF(os.path.join(os.path.dirname(__file__), '../urdf/plane.urdf'))


# load up Aliengo
quadruped = Aliengo(client, max_torque=40.0, fixed=False, kp=0.1, kd=1.0)
quadruped.reset_joint_positions(stochastic=False)

# change_dynamics(client, quadruped, plane)
# fix_feet_points(client, quadruped)

# visualize foot collision shapes
sphere_shape = client.createVisualShape(p.GEOM_SPHERE, 
                                        radius=0.0265, 
                                        rgbaColor=[0,2,0,1]) # 0.0265 is foot collision sphere radius 
sphere_ids = [0] * 4
for i in range(4):
    sphere_ids[i] = client.createMultiBody(baseVisualShapeIndex=sphere_shape)

# make the feet transparent
for link in quadruped.foot_links + quadruped.shin_links:
    client.changeVisualShape(quadruped.quadruped, link, rgbaColor=[0, 0, 0, 0.5])
update_collision_sphere_vis(client, quadruped, sphere_ids)


while True:
    with open('mocap.txt','r') as f:
        for line_num, line in enumerate(f): 
            if line_num%2 == 0: # Unitree runs this demo at 500 Hz. We run at 240 Hz, so double is close enough.
                action = np.array(line.split(',')[2:], dtype=np.float32)
                quadruped.set_joint_position_targets(action, true_positions=True)
            
            client.stepSimulation()
            update_collision_sphere_vis(client, quadruped, sphere_ids)
            # time.sleep(1/240.)
            

while True:
    client.stepSimulation()
    time.sleep(1/240.)
