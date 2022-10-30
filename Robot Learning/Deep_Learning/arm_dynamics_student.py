from arm_dynamics_base import ArmDynamicsBase
import numpy as np
import torch
from models import build_model


class ArmDynamicsStudent(ArmDynamicsBase):
    def init_model(self, model_path, num_links, time_step, device):
        # ---
        # Your code hoes here
        # Initialize the model loading the saved model from provided model_path
        self.model = build_model(num_links, time_step)
        self.model.load_state_dict(torch.load(model_path))

        # ---
        self.model_loaded = True

    def dynamics_step(self, state, action, dt):
        if self.model_loaded:
            # ---
            # Your code goes here
            # Use the loaded model to predict new state given the current state and action
            input_data = np.vstack((state, action)).T.astype(np.float32)
            new_state = self.model.forward( torch.tensor(input_data) ).clone().detach().numpy().T
            
            return new_state
            # ---
        else:
            return state

