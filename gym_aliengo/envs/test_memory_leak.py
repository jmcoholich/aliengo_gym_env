''' Script created to test the memory leak with p.calculateInverseKinematics2()

Command to run:
mprof run python test_memory_leak.py
mprof plot

When finished, to delete all generate logs:
mprof clean

Other requiements: 
memory-profiler https://pypi.org/project/memory-profiler/
    pip install memory-profiler
'''


import pybullet as p
import pybullet_data
import os

path = os.path.abspath(os.path.dirname(pybullet_data.__file__))
client = p.connect(p.GUI)
robot = p.loadSDF(os.path.join(path, 'kuka_iiwa/kuka_with_gripper.sdf'), physicsClientId=client)[0]

for _ in range(50_000): # increase number of iterations to show more memory leaking
    p.calculateInverseKinematics2(robot, [10], [[10]*3], physicsClientId=client) # function has memory leak
    # p.calculateInverseKinematics(robot, 10, [10]*3, physicsClientId=client) # run this command instead, no memory leak


