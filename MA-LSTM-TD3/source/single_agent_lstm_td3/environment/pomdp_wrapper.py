"""
Description:
    Here I try to implement a simplified POMDP wrapper for OpenAI Gymnasium MuJoCo environments.

    I am trying to implement it in a way that is the same or very similar to a LSTM-TD3 paper.

    https://arxiv.org/abs/2102.12344
    https://github.com/LinghengMeng/LSTM-TD3/blob/main/lstm_td3/env_wrapper/pomdp_wrapper.py

    Hopefully, I can implement it for the OpenAI Gymnasium environments,
    (1) "Ant-v4"
    (2) "HalfCheetah-v4"

    https://gymnasium.farama.org/environments/mujoco/

    The paper talks about 4 different POMDP scenarios.
    (1) POMDP-RemoveVelocity
        Velocities ore removed from the environment state/observation.

    (2) POMDP-Flickering
        With a fixed probability, all observation values are set to zero.
        This simulates a missed signal or faulty sensor in a robot.

    (3) POMDP-RandomNoise
        All the observation values will have noise added to them.

    (4) POMDP-RandomSensorMissing
        With a fixed probability, a random sensor will have its value set to zero.
        This is to simulate a faulty sensor on a robot.

Author:
    Jordan Cramer

Date:
    2023-07-23

References:
    LSTM-TD3 code

    * License: MIT License
    * Repo: https://github.com/LinghengMeng/LSTM-TD3
    * Repo: https://github.com/LinghengMeng/LSTM-TD3/blob/main/lstm_td3/env_wrapper/pomdp_wrapper.py

    The code that LinghengMeng originally used was from OpenAI SpinningUp.

    * License: MIT License
    * Repo: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/td3/
"""
from score_storage import ScoreStorage
import gymnasium as gym
import numpy as np


class POMDPWrapper(gym.ObservationWrapper):
    """
    This class will modify the observations/states received from the supported environments to create
    Partially Observable Markov Decision Processes (POMDPs).

    :param env_name:
        Supported Environments
        OpenAI Gymnasium MuJoCo "Ant-v4" (The paper use contact forces. Version 4 requires use_contact_forces=true.)
        https://gymnasium.farama.org/environments/mujoco/ant/
        OpenAI Gymnasium MuJoCo "HalfCheetah-v4"
        https://gymnasium.farama.org/environments/mujoco/half_cheetah/

    :param pomdp_type:
        1.  'remove_velocity': remove velocity related observation values from the observation/state array

        2.  'flickering': at each time step, with a certain probability (flicker_prob),
            return an array of zeros of the same size instead of the environment observation/state.

        3.  'random_noise': each value in the observation/state will have noise added to it,
            the noise is sampled from a normal distribution

        4.  'random_sensor_missing': at each time step, with a certain probability (random_sensor_missing_prob),
            a value in the environment observation/state will be set to 0 to simulate a sensor missing a value
    """
    def __init__(
            self,
            env_name,
            pomdp_type,
            flicker_prob=0.2,
            random_noise_sigma=0.1,  # standard deviation for the noise
            random_sensor_missing_prob=0.1
    ):
        # create the environments, including any specific settings
        if env_name == 'Ant-v4':
            super(gym.ObservationWrapper, self).__init__(gym.make(env_name, use_contact_forces=True))
        elif env_name == 'HalfCheetah-v4':
            super(gym.ObservationWrapper, self).__init__(gym.make(env_name))

        # record all the passed in arguments
        self.env_name = env_name
        self.pomdp_type = pomdp_type
        self.flicker_prob = flicker_prob
        self.random_noise_sigma = random_noise_sigma
        self.random_sensor_missing_prob = random_sensor_missing_prob

        # functionality to record scores during the episodes
        self.score_storage = ScoreStorage()

        # indices variable
        # for POMDPs where velocity values have been removed from the observation/state,
        # this variable records the indices/values that are still being used from the observation/state
        self.remain_obs_idx = None

        """
        Handle some of the POMDP set up here:
        
        pomdp_type == 'remove_velocity':
            figure out which observation/state indices/values are still being used from the original observation
        
        pomdp_type == 'flickering':
            this will be handled in the observation function (pass for now)
        
        pomdp_type == 'random_noise':
            this will be handled in the observation function (pass for now)
        
        pomdp_type == 'random_sensor_missing':
            this will be handled in the observation function (pass for now)
        """
        if pomdp_type == 'remove_velocity':
            self._remove_velocity(self.env_name)
        elif pomdp_type == 'flickering':
            pass
        elif pomdp_type == 'random_noise':
            pass
        elif pomdp_type == 'random_sensor_missing':
            pass
        else:
            raise ValueError(f"Error with the pomdp_type specified: {self.pomdp_type}")

    def step(self, action):
        """ I am overriding the step function to add reward recording functionality. """
        # call the original step function
        observation, reward, terminated, truncated, info = self.env.step(action)

        # override the observation using the observation function
        modified_observation = self.observation(observation)

        # record the score
        self.score_storage.add_reward(float(reward))

        return modified_observation, reward, terminated, truncated, info

    def observation(self, observation):
        """
        We are overriding the original environment observation/state returned at each time step
        with whatever is returned by this function (this is the purpose of an ObservationWrapper).

        :param observation: the original observation/state from the environment
        :return: the modified observation
        """
        if self.pomdp_type == 'remove_velocity':
            # return an observation with the velocity values removed
            return observation.flatten()[self.remain_obs_idx]
        elif self.pomdp_type == 'flickering':
            # with a certain probability, return an array of all zero values instead of the normal observation
            if np.random.rand() <= self.flicker_prob:
                return np.zeros(observation.shape)
            else:
                return observation.flatten()
        elif self.pomdp_type == 'random_noise':
            # return an observation where all the values have added noise
            return (observation + np.random.normal(0, self.random_noise_sigma, observation.shape)).flatten()
        elif self.pomdp_type == 'random_sensor_missing':
            # some of the observation values will randomly be set to zero with a certain probability
            observation[np.random.rand(len(observation)) <= self.random_sensor_missing_prob] = 0
            return observation.flatten()
        else:
            raise ValueError(f'Error pomdp_type was either wrong or not implemented: {self.pomdp_type}')

    def _remove_velocity(self, env_name):
        """
        This function will set the indices/values from the environment observation/state that will be used.
        All the indices will be included except the ones pertaining to velocity values.
        """
        # OpenAI Gymnasium MuJoCo Environments
        # 'Ant-v4'
        # https://gymnasium.farama.org/environments/mujoco/ant/
        # 'HalfCheetah-v4'
        # https://gymnasium.farama.org/environments/mujoco/half_cheetah/

        if env_name == 'Ant-v4':
            # 0-12: coordinates, 13-26: velocities, 27-110: contact forces (env setting: use_contact_forces=True)
            self.remain_obs_idx = list(np.arange(0, 13)) + list(np.arange(27, 111))
        elif env_name == 'HalfCheetah-v4':
            self.remain_obs_idx = np.arange(0, 8)
        else:
            raise ValueError(f'POMDP for {env_name} is not defined.')

        # redefine the observation space
        # since the observation space has changed due to using a subset of the indices/values
        low = np.array([-np.inf for _ in range(len(self.remain_obs_idx))], dtype="float32")
        high = np.array([np.inf for _ in range(len(self.remain_obs_idx))], dtype="float32")
        self.observation_space = gym.spaces.Box(low, high)


def main():
    """ Quickly testing out the wrapper code down below to make sure it works. """
    # checking the OpenAI Gymnasium MuJoCo 'Ant-v4' wrapped environment
    print('------- Testing Wrapped Ant-v4 -------')
    wrapped_ant = POMDPWrapper('Ant-v4', pomdp_type='remove_velocity')
    observation, info = wrapped_ant.reset()
    print('------- Action Space -------')
    print(wrapped_ant.action_space)
    print('------- Observation Space Shape -------')
    print(wrapped_ant.observation_space.shape)
    print('------- Observation Space -------')
    print(wrapped_ant.observation_space)
    print('------- Observation Shape -------')
    print(observation.shape)
    print('------- Observation -------')
    print(observation)
    print('------- Sample Reward -------')
    observation, reward, terminated, truncated, info = wrapped_ant.step(wrapped_ant.action_space.sample())
    print('Reward Type:', type(reward))
    print(reward)
    print()

    # checking the OpenAI Gymnasium MuJoCo 'HalfCheetah-v4' wrapped environment
    print('------- Testing Wrapped HalfCheetah-v4 -------')
    wrapped_cheetah = POMDPWrapper('HalfCheetah-v4', pomdp_type='remove_velocity')
    observation, info = wrapped_cheetah.reset()
    print('------- Action Space -------')
    print(wrapped_cheetah.action_space)
    print('------- Observation Space Shape -------')
    print(wrapped_cheetah.observation_space.shape)
    print('------- Observation Space -------')
    print(wrapped_cheetah.observation_space)
    print('------- Observation Shape -------')
    print(observation.shape)
    print('------- Observation -------')
    print(observation)
    print('------- Sample Reward -------')
    observation, reward, terminated, truncated, info = wrapped_cheetah.step(wrapped_cheetah.action_space.sample())
    print('Reward Type:', type(reward))
    print(reward)
    print()

    # unchanged 'Ant-v4'
    original_ant = gym.make('Ant-v4', use_contact_forces=True)
    # observation, info = env.reset()

    # unchanged 'HalfCheetah-v4'
    original_cheetah = gym.make('HalfCheetah-v4')
    # observation, info = env.reset()


if __name__ == '__main__':
    """ Quickly testing out the wrapper code down below to make sure it works. """
    main()
