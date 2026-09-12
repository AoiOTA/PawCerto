# Copyright (c) 2022 MIT Improbable AI Lab. See ../../LICENSE.
# Fixed original sensor; simulator math imports replaced by tensor algebra.
from .sensor import Sensor
from ..math_utils import quat_apply_yaw
from ..torch_math import *

class EeGripperTargetPositionSensor(Sensor): 
    def __init__(self, env, attached_robot_asset=None, delay=0):
        super().__init__(env)
        self.env = env

    def get_observation(self, env_ids = None):
        return self.env.commands[:, 15:18]
    
    def get_noise_vec(self):
        import torch
        return torch.zeros(3, device=self.env.device)
    
    def get_dim(self):
        return 3
