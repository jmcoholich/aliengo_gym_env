from gym_aliengo.envs import aliengo_env
from pybullet_utils import bullet_client as bc
import pybullet as p
from os.path import dirname, join

class AliengoEnvParent(aliengo_env.AliengoEnv):

    def __init__(self, **kwargs):
        
        # make 'hutter_pmtg' the default mode for all environments with terrain (which are children of this class)
        if not 'env_mode' in kwargs:
            kwargs['env_mode'] = 'hutter_pmtg'

        # Make flat ground as False (enable terrain scan) for envs that have terrain
        if not 'flat_ground' in kwargs:
            kwargs['flat_ground'] = False
        print(kwargs)

        super().__init__(**kwargs)

        self.fake_client = bc.BulletClient(connection_mode=p.DIRECT) 
        if self.fake_client == -1:
            raise RuntimeError('Pybullet could not connect to physics client')
        # heightmap param dict 
        self.heightmap_params = {'length': 1.25, # assumes square 
                            'robot_position': 0.5, # distance of robot base origin from back edge of height map
                            'grid_spacing': 0.125}
        assert self.heightmap_params['length'] % self.heightmap_params['grid_spacing'] == 0
        self.grid_len = int(self.heightmap_params['length']/self.heightmap_params['grid_spacing']) + 1


    def _hard_reset(self):
        '''Resets the simulation, sets simulation parameters and loads plane into both clients.'''

        self.client.resetSimulation()
        self.fake_client.resetSimulation() 
    
        self.client.setTimeStep(1/240.)
        self.client.setGravity(0,0,-9.8)
        self.client.setRealTimeSimulation(self.realTime) # this has no effect in DIRECT mode, only GUI mode

        self.plane = self.client.loadURDF(join(dirname(__file__), '../urdf/plane.urdf'))
        self.fake_plane = self.fake_client.loadURDF(join(dirname(__file__), '../urdf/plane.urdf'))
    

