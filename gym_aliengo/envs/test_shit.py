import pybullet
import gym
import time

env = gym.make('gym_aliengo:Aliengo-v0', render=True)

while True: 
    env.reset()
    # time.sleep(0.1)