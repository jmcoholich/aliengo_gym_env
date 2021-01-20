import pybullet as p
import gym
import time
import os
import pybullet_envs.minitaur.envs.env_randomizers.minitaur_terrain_randomizer as asdf
import pybullet_envs.minitaur.envs.minitaur_gym_env as ggez
from PIL import Image
import numpy as np
from gym_aliengo.envs import aliengo
from pybullet_utils import bullet_client as bc


client = bc.BulletClient(connection_mode=p.DIRECT)
quadruped = aliengo.Aliengo(client)

# test speeds of looking up joint info vs storing it myself 
# lets say I call the function once per env.step(), and a run has 100M steps
start = time.time()
for i in range(100):
    info = client.getJointStates(quadruped.quadruped, quadruped.motor_joint_indices)
end = time.time()
elapsed = end - start
print('\n\nElapsed Time: {} min for 100M steps of env\n OR {} percent of runtime'.format(elapsed * 1e6/60.,
                                                             elapsed * 1e6/3600.0/40.0 * 100  ))

# basically just don't worry about this lol 







