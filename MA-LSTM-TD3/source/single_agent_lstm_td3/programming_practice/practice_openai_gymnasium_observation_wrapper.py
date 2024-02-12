"""
Description:
    Here I am practicing using observation wrappers for OpenAI Gymnasium Classical Control and MuJoCo environments.

    ObservationWrapper:
        OpenAI Gymnasium Environments return state/observation information at each time step.
        Usually, this information is in the form of a numpy.ndarray of floating point values.

        If we want to change the floating point values (state/observation) returned,
        such as scaling them, adding noise, or even removing values from the array,
        then we can use a wrapper.

    I will concatenate the original value array with the modified array, so it is easier to check the modifications.
    [[original array]
    [modified array]]

Author:
    Jordan Cramer

Date:
    2023-07-23
"""
import gymnasium as gym
import numpy as np


class ScalingObservationWrapper(gym.ObservationWrapper):
    """
    This is a wrapper class that works exactly like an OpenAI Gymnasium environment,
    except it modifies the observations/states that are returned during time steps of the environment.
    """
    def __init__(self, environment_name: str):
        """
        :param environment_name: The OpenAI Gymnasium environment to be created. Example: "Ant-v4" "CartPole-v1"
        """
        # call the base class constructor to set up the base class code
        super(ScalingObservationWrapper, self).__init__(gym.make(environment_name))

    def observation(self, observation):
        """
        We can modify the original observation here.
        """
        # scaled the observation up by 2
        modified_observation = observation * 2

        # vertical concatenation
        modified_observation = np.vstack((observation, modified_observation))

        return modified_observation


class NoiseObservationWrapper(gym.ObservationWrapper):
    """
    Adds noise sampled from a normal distribution to each float in the observation/state array.
    """
    def __init__(self, environment_name: str):
        super(NoiseObservationWrapper, self).__init__(gym.make(environment_name))
        self.random_noise_sigma = 0.01

    def observation(self, observation):
        # here we are sampling noise from a normal distribution with mean=0 and std=self.random_noise_sigma
        # this noise array, which will be the same shape as the observation, is added to the observation
        noise_array = np.random.normal(0, self.random_noise_sigma, observation.shape)
        modified_observation = (observation + noise_array).flatten()

        # vertical concatenation
        modified_observation = np.vstack((observation, noise_array, modified_observation))

        return modified_observation


class RemoveObservationWrapper(gym.ObservationWrapper):
    """
    Removes a value from the observation/state of the environment.
    """
    def __init__(self, environment_name: str):
        super(RemoveObservationWrapper, self).__init__(gym.make(environment_name))

    def observation(self, observation):
        # here I am going to assume the 'CartPole-v1' environment
        # I will remove the 4th observation which is 'Pole Angular Velocity'
        # the final shape of the observation would be (4,) ---> (3,)
        indices = [0, 1, 2]  # 3 will no longer be included

        # vertical view
        # [
        #   [Original]
        #   [Removed Observation]
        # ]
        cartpole_num_of_observation_values = 4
        modified_observation = np.zeros((2, cartpole_num_of_observation_values), dtype=float)
        modified_observation[0, :] = observation
        modified_observation[1, 0:3] = observation[indices]

        return modified_observation


def main():
    """ Perform a quick test to see if my code is working and make sure I understand ObservationWrapper correctly. """
    # I create different environments to see how the wrappers are working
    #
    # (1) Environment with no changes
    # (2) Environment wrapper that scales the observation
    # (3) Environment wrapper that adds noise to the observation
    # (4) Environment wrapper that removes a value from the observation array
    environment_name = 'CartPole-v1'
    original_env = gym.make(environment_name)
    scaling_env = ScalingObservationWrapper(environment_name)
    noise_env = NoiseObservationWrapper(environment_name)
    remove_env = RemoveObservationWrapper(environment_name)

    # try resetting all the environments and print the observations to check them
    seed_value = 100
    original_observation, info = original_env.reset(seed=seed_value)
    scaling_observation, _ = scaling_env.reset(seed=seed_value)
    noise_observation, _ = noise_env.reset(seed=seed_value)
    remove_observation, _ = remove_env.reset(seed=seed_value)

    # let's print the observations and see what they look like
    print('------- Original Observation -------')
    print(original_observation)

    print('------- Scaling Observation -------')
    print('[\n\t[Original]\n\t[2x Scaled]\n]')
    print(scaling_observation)

    print('-------Noise Observation -------')
    print('[\n\t[Original]\n\t[Noise]\n\t[Original + Noise]\n]')
    print(noise_observation)

    print('------- Remove Observation -------')
    print('[\n\t[Original]\n\t[Removed Observation]\n]')
    print(remove_observation)


if __name__ == '__main__':
    main()
