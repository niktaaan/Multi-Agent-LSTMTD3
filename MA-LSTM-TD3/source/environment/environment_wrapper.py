# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 09:34:02 2023

@author: m.kiakojouri
"""

import numpy as np
import gymnasium as gym

class Environment_Wrapper(gym.ObservationWrapper):
    def __init__(self, env, pomdp_type, n_missing_sensor=2, noise_std=0.2):
        super().__init__(env)
        self.pomdp_type = pomdp_type
        self.n_missing_sensor = n_missing_sensor
        self.noise_std = noise_std

    def observation(self, obs):
        if self.pomdp_type == "random_sensor_missing":
            for key, value in obs.items():
                if self.n_missing_sensor < len(value):
                    random_indices = np.random.choice(len(value), self.n_missing_sensor, replace=False)
                    value[random_indices] = 0  # Set the selected indices to 0

        elif self.pomdp_type == 'landmark_missing_position':
            for key, value in obs.items():
                indices = list(range(4, 6))
                value[indices] = 0  # Set the selected indices to 0

        elif self.pomdp_type == 'random_noise':
            for key, value in obs.items():
                value = value + np.random.normal(0, self.noise_std, value.shape)

        return obs
