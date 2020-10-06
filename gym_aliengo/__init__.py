from gym.envs.registration import register

register(
    id='aliengo-v0',
    entry_point='gym_aliengo.envs:AliengoEnv',
)


register(
    id='MinitaurBulletEnv_PathFollow-v0',
    entry_point='gym_aliengo.envs:MinitaurBulletEnv_PathFollow',
)
# register(
#     id='aliengo-extrahard-v0',
#     entry_point='gym_aliengo.envs:AliengoExtraHardEnv',
# )
register(
    id='MinitaurBulletEnv_Friction-v0',
    entry_point='gym_aliengo.envs:MinitaurBulletEnv_Friction',
)

