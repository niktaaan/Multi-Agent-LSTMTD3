"""
Description:
    A replay buffer that can also store variable length histories of observation-action pairs.
    For the LSTM-TD3 reinforcement learning agent.

Example:
        A history of length 5
        h_5 = [(o1,a1),(o2,a2),(o3,a3),(o4,a4),(o5,a5)]

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
import torch
import numpy as np


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for agents.
    """
    def __init__(
            self,
            observation_shape,
            action_shape,
            buffer_size: int
    ):
        """ Initialize the replay buffer. """

        # save passed in arguments
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.buffer_size = buffer_size

        # create storage for the replay buffer
        self.observation_memory = np.zeros((buffer_size, observation_shape), dtype=np.float32)
        self.next_observation_memory = np.zeros((buffer_size, observation_shape), dtype=np.float32)
        self.action_memory = np.zeros((buffer_size, action_shape), dtype=np.float32)
        self.reward_memory = np.zeros(buffer_size, dtype=np.float32)
        self.terminal_memory = np.zeros(buffer_size, dtype=np.float32)

        # keep track of the current index to store experiences at, and the current size (number of experiences stored)
        self.memory_index = 0
        self.size = 0

    def store_experience(self, observation, action, reward, next_observation, termination):
        """
        Store an experience in the replay buffer.

        Args:
            observation: The current environment observation.

            action: The action taken by the agent given the current environment observation and its policy.

            reward: The reward obtained by the agent after taking the action in the environment.

            next_observation: The next state/observation of the environment after the agent took an action.

            termination: A flag signalling that the agent has reached a terminal state in the environment. (Episode Ended) This should only be true if the agent's action caused the episode to end. Truncation (ending an episode artificially after a certain number of time steps) should set this to False.
        """
        # store all the experience information in the buffer at the current memory index
        self.observation_memory[self.memory_index] = observation
        self.action_memory[self.memory_index] = action
        self.reward_memory[self.memory_index] = reward
        self.next_observation_memory[self.memory_index] = list(next_observation)
        self.terminal_memory[self.memory_index] = termination

        # calculate the next replay buffer index to store an experience
        self.memory_index = (self.memory_index + 1) % self.buffer_size

        # the number of currently stored experiences is either,
        # (1) the number of stored memories (if it is smaller than the max buffer size)
        # (2) the max buffer size (when the replay buffer is completely filled)
        self.size = min(self.size + 1, self.buffer_size)

    def sample_buffer(self, batch_size: int = 32) -> dict:
        """
        Sample a batch of experiences uniformly from the replay buffer.
        These are simple experiences that do not include observation-action pair histories.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            A dictionary containing the batch of experiences.
        """
        # obtain random indices for experiences to sample
        indices = np.random.randint(0, self.size, size=batch_size)

        # sample the batch of experiences using the indices
        batch = dict(
            observations=self.observation_memory[indices],
            next_observations=self.next_observation_memory[indices],
            actions=self.action_memory[indices],
            rewards=self.reward_memory[indices],
            terminations=self.terminal_memory[indices]
        )

        # return the batch of simple experiences
        return {key: torch.as_tensor(value, dtype=torch.float32) for key, value in batch.items()}

    def sample_buffer_with_history(self, batch_size: int = 32, max_history_length: int = 100) -> dict:
        """
        Sample a batch of experiences uniformly from the replay buffer.
        These include observation-action pair histories.

        Args:
            batch_size (int):
                The number of experiences to sample.

            max_history_length (int):
                The length/number of experiences in the history before current experience.

        Returns:
            A dictionary containing the batch of experiences with observation-action pair histories.
        """
        # obtain random indices for experiences to sample
        indices = np.random.randint(max_history_length, self.size, size=batch_size)

        # get the batch where history length is 0
        if max_history_length == 0:
            history_observations = np.zeros([batch_size, 1, self.observation_shape])
            history_actions = np.zeros([batch_size, 1, self.action_shape])
            history_next_observations = np.zeros([batch_size, 1, self.observation_shape])
            history_next_actions = np.zeros([batch_size, 1, self.action_shape])
            history_observations_length = np.zeros(batch_size)
            history_next_observations_length = np.zeros(batch_size)

        # get the batch where history length is not 0 (using observation-action histories)
        else:
            history_observations = np.zeros([batch_size, max_history_length, self.observation_shape])
            history_actions = np.zeros([batch_size, max_history_length, self.action_shape])
            history_next_observations = np.zeros([batch_size, max_history_length, self.observation_shape])
            history_next_actions = np.zeros([batch_size, max_history_length, self.action_shape])
            history_observations_length = max_history_length * np.ones(batch_size)
            history_next_observations_length = max_history_length * np.ones(batch_size)

            # extract history experiences before sampled index
            # a history includes all the observation-action pairs before the index (up to the max_history_length)
            for i, index in enumerate(indices):

                # determine the starting index in the buffer for the history
                history_start_index = index - max_history_length
                if history_start_index < 0:
                    history_start_index = 0

                # if the history terminates before the last experience
                # (not including a terminal flag in index)
                # start from the index next to the termination
                if len(np.where(self.terminal_memory[history_start_index:index] == 1)[0]) != 0:
                    history_start_index = history_start_index + (np.where(self.terminal_memory[history_start_index:index] == 1)[0][-1]) + 1
                history_segment_length = index - history_start_index
                history_observations_length[i] = history_segment_length
                history_observations[i, :history_segment_length, :] = self.observation_memory[history_start_index:index]
                history_actions[i, :history_segment_length, :] = self.action_memory[history_start_index:index]

                # if the first experience of an episode is sampled,
                # the history lengths are different for observations and next_observations.
                if history_segment_length == 0:
                    history_next_observations_length[i] = 1
                else:
                    history_next_observations_length[i] = history_segment_length
                history_next_observations[i, :history_segment_length, :] = self.next_observation_memory[history_start_index:index]
                history_next_actions[i, :history_segment_length, :] = self.action_memory[history_start_index+1:index+1]

        # sample the batch of experiences using the indices
        batch = dict(
            observations=self.observation_memory[indices],
            next_observations=self.next_observation_memory[indices],
            actions=self.action_memory[indices],
            rewards=self.reward_memory[indices],
            terminations=self.terminal_memory[indices],
            history_observations=history_observations,
            history_actions=history_actions,
            history_next_observations=history_next_observations,
            history_next_actions=history_next_actions,
            history_observations_length=history_observations_length,
            history_next_observations_length=history_next_observations_length
        )

        # return: the batch of experiences with observation-action pair histories
        # also: convert from numpy arrays to PyTorch tensors
        return {key: torch.as_tensor(value, dtype=torch.float32) for key, value in batch.items()}
