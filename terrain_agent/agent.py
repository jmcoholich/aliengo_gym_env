import torch
from torch import nn

class TerrainAgent(nn.Module):
    def __init__(self, means, stds):
        super(TerrainAgent, self).__init__()
        self.means = means
        self.stds = stds
        
        # the image size is going to be 65 x 65
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 2, 7) 
        self.conv2 = nn.Conv2d(2, 4, 5) 
        self.conv3 = nn.Conv2d(4, 8, 3) 
        self.encoder_linear = nn.Linear(392, 64)
        self.linear1 = nn.Linear(77, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 3) # output is xyz relative to robot torso center.
        # TODO make sure all inputs/ and outputs and relative to each other, not in absolute coordinates or anything

    def forward(self, foot_positions, foot, heightmap):
        # first normalize
        EPS = 1e-2
        foot_positions = (foot_positions - self.means[0])/(self.stds[0] + EPS) 
        foot = (foot - self.means[1])/(self.stds[1] + EPS)
        heightmap = (heightmap - self.means[2])/(self.stds[2] + EPS)
        
        assert foot_positions.shape[0] == foot.shape[0] == heightmap.shape[0] # batch sizes are all the same
        n = foot_positions.shape[0]
        encoded_pic = self.cnn_encoder(heightmap)
        x = torch.cat((encoded_pic, foot_positions.reshape(n, 12), foot), dim=1) # len is 64 + 12 + 1 = 77
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x


    def cnn_encoder(self, x): # x begins as 1 x81 x 81
        n = x.shape[0]
        x = self.relu(self.conv1(x)) # 2 x 75 x 75
        x = self.pool(x) # 2 x 37 x 37
        x = self.relu(self.conv2(x)) # 4 x 33 x 33
        x = self.pool(x) # 4 x 16 x 16
        x = self.relu(self.conv3(x)) # 8 x 14 x 14
        x = self.pool(x) # 8 x 7 x 7
        x = self.encoder_linear(x.reshape(n, 8 * 49)) # 64
        assert len(x.shape) == 2
        return x # 64