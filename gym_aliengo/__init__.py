from gym.envs.registration import register

register(
    id='aliengo-v0',
    entry_point='gym_aliengo.envs:AliengoEnv',
)
# register(
#     id='aliengo-extrahard-v0',
#     entry_point='gym_aliengo.envs:AliengoExtraHardEnv',
# )
