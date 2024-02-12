"""
Description:
    A wrapper class for creating MDP OpenAI Gymnasium MuJoCo environments.

Author:
    Jordan Cramer

Date:
    2023-07-23
"""
from score_storage import ScoreStorage
import gymnasium as gym


class MDPWrapper(gym.Wrapper):
    """
    Simple functionality to record scores is being added.
    """

    def __init__(
            self,
            env_name,
            animate_flag=False
    ):
        """
        :param env_name:
            provide the OpenAI Gymnasium Environment name
        :param animate_flag:
            set to true if you would like the environment episodes to be animated and displayed
        """
        # create the environments, including any specific settings
        if env_name == 'Ant-v4':
            if animate_flag:
                super().__init__(gym.make(env_name, use_contact_forces=True, render_mode='human'))
            else:
                super().__init__(gym.make(env_name, use_contact_forces=True))
        elif env_name == 'HalfCheetah-v4':
            if animate_flag:
                super().__init__(gym.make(env_name, render_mode='human'))
            else:
                super().__init__(gym.make(env_name))
        else:
            raise ValueError(f'Error with the environment \'{env_name}\'. Is it supported by the wrapper or misspelled?')

        # record the passed in arguments
        self.env_name = env_name

        # functionality to record scores during the episodes
        self.score_storage = ScoreStorage()

    def step(self, action):
        """ I am overriding the step function to add reward recording functionality. """
        # call the original step function
        observation, reward, terminated, truncated, info = self.env.step(action)

        # record the score
        self.score_storage.add_reward(float(reward))

        return observation, reward, terminated, truncated, info


""" Quickly testing out the wrapper code down below to make sure it works. """
if __name__ == '__main__':
    wrapped_env = MDPWrapper('Ant-v4')
