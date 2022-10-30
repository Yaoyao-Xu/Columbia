import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, env):
        super(QNetwork, self).__init__()
        #--------- YOUR CODE HERE --------------

        self.Q_model = self.get_layers()

        #---------------------------------------

    def get_layers(self):
        layers = nn.Sequential(
        nn.Linear(6, 64),
        nn.ReLU(),
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 37)
      )

        return layers

    def forward(self, x, device):
        #--------- YOUR CODE HERE --------------
        x = torch.Tensor(x, device = device)
        out = self.Q_model(x)
        return out
        #---------------------------------------
        
    def select_discrete_action(self, obs, device):
        # Put the observation through the network to estimate q values for all possible discrete actions
        est_q_vals = self.forward(obs.reshape((1,) + obs.shape), device)
        # Choose the discrete action with the highest estimated q value
        discrete_action = torch.argmax(est_q_vals, dim=1).tolist()[0]
        return discrete_action
                
    def action_discrete_to_continuous(self, discrete_action):
        #--------- YOUR CODE HERE --------------


        action_dict = { 0: [0,0], 
                    1:[0.2, 0.1] ,    2:[-0.2,-0.1],    3:[0.2, -0.1],
                    4: [-0.2, 0.1],   5:[0.4, 0.2],     6:[-0.4, -0.2], 
                    7:[0.4, -0.2],    8:[-0.4,0.2],     9:[0.5,0.3], 
                    10:[-0.5, -0.3],  11:[0.5, -0.3],   12:[-0.5,0.3], 
                    13:[0.8, 0.4],    14:[-0.8, -0.4],  15:[0.8, -0.4],
                    16:[-0.8, 0.4],   17:[1.0, 0.6],    18:[-1.0, -0.6], 
                    19:[1.0, -0.6],   20: [-1.0, 0.6],  21:[1.2, 0.6], 
                    22:[-1.2, -0.6],  23:[1.2, -0.6],   24:[-1.2, 0.6], 
                    25:[1.5, 0.8],    26:[-1.5, -0.8],  27:[1.5, -0.8],
                    28:[-1.5, 0.8],   29:[1.6,1.0],     30:[-1.6,-1.0],
                    31:[1.6, -1.0],   32:[-1.6, 1.0],   33:[1,1],
                    34:[-1,-1],       35:[1, -1],       36:[-1,1]
        }

        continuous_action = action_dict[discrete_action]
        con_action = np.array(continuous_action).reshape(2,1)
        return con_action
        #---------------------------------------
