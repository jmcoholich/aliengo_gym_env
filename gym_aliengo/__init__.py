from gym.envs.registration import register

register(
    id='Aliengo-v0',
    entry_point='gym_aliengo.envs:AliengoEnv',
)

register(
    id='AliengoSteppingStones-v0',
    entry_point='gym_aliengo.envs:AliengoSteppingStones',
)

register(
    id='AliengoHills-v0',
    entry_point='gym_aliengo.envs:AliengoHills',
)

register(
    id='AliengoSteps-v0',
    entry_point='gym_aliengo.envs:AliengoSteps',
)

register(
    id='AliengoStairs-v0',
    entry_point='gym_aliengo.envs:AliengoStairs',
)

register(
    id='AliengoTrotInPlace-v0',
    entry_point='gym_aliengo.envs:AliengoTrotInPlace',
)

register(
    id='FootstepParam-v0',
    entry_point='gym_aliengo.envs:FootstepParam',
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

