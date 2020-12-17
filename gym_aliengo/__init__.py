from gym.envs.registration import register

register(
    id='Aliengo-v0',
    entry_point='gym_aliengo.envs:AliengoEnv',
)

register(
    id='AliengoSteppingStones-v0',
    entry_point='gym_aliengo.envs:AliengoSteppingStones',
)


# register(
#     id='MinitaurBulletEnv_PathFollow-v0',
#     entry_point='gym_aliengo.envs:MinitaurBulletEnv_PathFollow',
# )
# # register(
# #     id='aliengo-extrahard-v0',
# #     entry_point='gym_aliengo.envs:AliengoExtraHardEnv',
# # )
# register(
#     id='MinitaurBulletEnv_Friction-v0',
#     entry_point='gym_aliengo.envs:MinitaurBulletEnv_Friction',
# )

