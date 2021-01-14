''' Script created to test the memory leak with p.calculateInverseKinematics2()

TODO reproduce the memory leak issue and submit a git issue

Command to run:
mprof run python test_memory_leak.py

mprof plot


'''
import gym
import pybullet as p
import numpy as np

env = gym.make('gym_aliengo:Aliengo-v0', use_pmtg=True)
env.reset()


action = np.random.rand(16)
for _ in range(50_000):
    # env.reset()
    env.step(action) # the issue is with env.step(). ONLY when PMTG is enabled


