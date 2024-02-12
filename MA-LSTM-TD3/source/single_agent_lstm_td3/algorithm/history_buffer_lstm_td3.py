"""
Description:
    A buffer for storing recent observation-action histories.
    For the LSTM-TD3 reinforcement learning agent.

Author:
    Jordan Cramer

Date:
    2023-08-08

References:
    LSTM-TD3 code

    * License: MIT License
    * Repo: https://github.com/LinghengMeng/LSTM-TD3

    The code that LinghengMeng originally used was from OpenAI SpinningUp.

    * License: MIT License
    * Repo: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/td3/
"""
import numpy as np


class HistoryBuffer:
    """
    A buffer for storing recent observation-action histories.
    For the LSTM-TD3 reinforcement learning agent.
    """
    def __init__(
            self,
            observation_shape,
            action_shape,
            max_history_length: int
    ):
        """
        Args:
            observation_shape:
            action_shape:
            max_history_length (int):
        """
        # save important variable values
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.max_history_length = max_history_length

        # prepare history buffers to store observation-action histories of the correct length and size
        self.observation_buffer = None
        self.action_buffer = None
        self.observation_buffer_length = None

    def initialize(self, starting_observation):
        """
        At the beginning of each new episode, the history buffer must be initialized.

        Args:
            starting_observation: The starting observation from the environment after the environment has been reset.
        """
        # if self.max_history_length > 0 (if using observation-action histories)
        if self.max_history_length > 0:
            self.observation_buffer = np.zeros([self.max_history_length, self.observation_shape])
            self.action_buffer = np.zeros([self.max_history_length, self.action_shape])
            self.observation_buffer[0, :] = starting_observation
            self.observation_buffer_length = 0

        # else (not using observation-action histories)
        else:
            self.observation_buffer = np.zeros([1, self.observation_shape])
            self.action_buffer = np.zeros([1, self.action_shape])
            self.observation_buffer_length = 0

    def add_observation_action(self, observation, action):
        """
        Every time there is a new observation and action, add it to the history buffer.

        Args:
            observation: The observation from the environment to be added to the history buffer.

            action: The action of the agent to be added to the history buffer.
        """

        """ add the most recent observation and action to the current history """

        # if self.max_history_length != 0 (if using observation-action histories)
        if self.max_history_length != 0:
            if self.observation_buffer_length == self.max_history_length:
                self.observation_buffer[:self.max_history_length - 1] = self.observation_buffer[1:]
                self.action_buffer[:self.max_history_length - 1] = self.action_buffer[1:]
                self.observation_buffer[self.max_history_length - 1] = list(observation)
                self.action_buffer[self.max_history_length - 1] = list(action)
            else:
                self.observation_buffer[self.observation_buffer_length] = list(observation)
                self.action_buffer[self.observation_buffer_length] = list(action)
                self.observation_buffer_length += 1
