# Copyright (c) 2022 MIT Improbable AI Lab. See ../../LICENSE.
# Fixed original sensor; simulator math imports replaced by tensor algebra.
class Sensor:
    def __init__(self, env):
        self.env = env

    def get_observation(self):
        raise NotImplementedError

    def get_noise_vec(self):
        raise NotImplementedError
    
    def get_dim(self):
        raise NotImplementedError
