import pybullet
import gym
import time
import os
import pybullet_envs.minitaur.envs.env_randomizers.minitaur_terrain_randomizer as asdf
import pybullet_envs.minitaur.envs.minitaur_gym_env as ggez
from PIL import Image



# # visualize loading terrain9735.obj mesh that Erwin Coumans provided. 

# env = ggez.MinitaurGymEnv(render=True)
# terrain = asdf.MinitaurTerrainRandomizer(
#                 terrain_type=asdf.TerrainType.RANDOM_BLOCKS,
#                 mesh_filename=os.path.join(os.path.dirname(__file__), '../meshes/terrain9735.obj'),
#                 mesh_scale=[1.0, 1.0, 10.0])

# while True:
#     terrain.randomize_env(env)
#     time.sleep(1)


# Generate mesh using Perlin noise

from noise import pnoise2
import numpy as np

# need to put a flat spot in the middle for the quadruped to spawn.
# all units in meters
hills_length = 100
hills_width = 100

# Perlin Noise parameters
scale = 100.0
octaves = 1
persistence = 0.5
lacunarity = 2.0

for k in range(10):
    vertices = np.zeros((hills_width + 1, hills_length + 1))
    base = np.random.randint(1000)
    print(base)
    for i in range(hills_width + 1):
        for j in range(hills_length + 1):
            vertices[i, j] = pnoise2(i/scale,
                                        j/scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        repeatx=hills_width + 1,
                                        repeaty=hills_length + 1,
                                        base=base) # base is the seed
    # breakpoint()
    Image.fromarray(((np.interp(vertices, (vertices.min(), vertices.max()), (0, 255.0)) > 128) * 255).astype('uint8'),
                     'L').show()
# vertices = np.interp(vertices, (vertices.min(), vertices.max()), (0, 1.0))

# # ramp down the n meters, so the robot can walk onto the hills terrain
# n = 1
# for i in range(n):
#     vertices[:, i] *= float(i)/n

# vertices = vertices * 1

# with open('../meshes/generated_hills.obj','w') as f:
#     f.write('o Generated_Hills_Terrain\n')
#     # write vertices
#     for i in range(hills_length + 1):
#         for j in range(hills_width + 1):
#             f.write('v  {}   {}   {}\n'.format(i, j, vertices[i, j]))
    
#     # write faces 
#     n_triangles = hills_length * hills_width * 2
#     for i in range(hills_length):
#         for j in range(hills_width):
#             # bottom left triangle
#             f.write('f  {}   {}   {}\n'.format((hills_width + 1)*i + j+1, 
#                                                 (hills_width + 1)*i + j+2, 
#                                                 (hills_width + 1)*(i+1) + j+1)) 

#             # top right triangle
#             f.write('f  {}   {}   {}\n'.format((hills_width + 1)*(i+1) + j+2, 
#                                                 (hills_width + 1)*(i+1) + j+1, 
#                                                 (hills_width + 1)*i + j+2)) 

#             # repeat, making faces double-sided
#             f.write('f  {}   {}   {}\n'.format((hills_width + 1)*i + j+2, 
#                                                 (hills_width + 1)*i + j+1, 
#                                                 (hills_width + 1)*(i+1) + j+1)) 

#             # top right triangle
#             f.write('f  {}   {}   {}\n'.format((hills_width + 1)*(i+1) + j+1, 
#                                                 (hills_width + 1)*(i+1) + j+2, 
#                                                 (hills_width + 1)*i + j+2)) 





# shape = (1024,1024)
# scale = 100.0
# octaves = 1
# persistence = 0.5
# lacunarity = 2.0

# world = np.zeros(shape)
# for i in range(shape[0]):
#     for j in range(shape[1]):
#         world[i][j] = pnoise2(i/scale, 
#                                     j/scale, 
#                                     octaves=octaves, 
#                                     persistence=persistence, 
#                                     lacunarity=lacunarity, 
#                                     repeatx=1024, 
#                                     repeaty=1024, 
#                                     base=0)


# world = np.interp(world, (world.min(), world.max()), (0, 255)).astype('uint8')
# Image.fromarray(world, 'L').show()
