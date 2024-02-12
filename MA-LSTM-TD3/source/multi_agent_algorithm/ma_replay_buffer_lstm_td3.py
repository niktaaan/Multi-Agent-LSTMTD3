import numpy as np
import torch
from source.multi_agent_algorithm.replay_buffer_lstm_td3 import ReplayBufferLSTMTD3


class MultiAgentReplayBufferLSTMTD3:
    """
    An experience replay buffer for multiple LSTM-TD3 agents in a MARL algorithm.
    For each lstm-td3 agent, a single agent replay buffer will be created.
    When sampling, histories of (observation,action) pairs will be sampled and concatenated as needed.
    """
    def __init__(
            self,
            number_of_agents: int,
            buffer_size: int,
            observation_sizes: list[int],
            action_sizes: list[int],
            batch_size: int,
            max_history_length: int
    ):
        # save all the passed in argument values
        self.number_of_agents = number_of_agents
        self.buffer_size = buffer_size
        self.observation_sizes = observation_sizes
        self.action_sizes = action_sizes
        self.batch_size = batch_size
        self.max_history_length = max_history_length

        # calculate the critic state size (concatenated observations and actions of all agents)
        self.critic_state_size: int = sum(self.observation_sizes) + sum(self.action_sizes)

        # a counter for keeping track of how many transitions/experiences/memories are stored in the buffer
        self.memory_counter = 0

        # create a replay buffer for each agent
        self.buffers: list[ReplayBufferLSTMTD3] = [
            ReplayBufferLSTMTD3(
                observation_size=self.observation_sizes[i],
                action_size=self.action_sizes[i],
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                max_history_length=self.max_history_length
            )
            for i in range(self.number_of_agents)
        ]

    def episode_reset(self, starting_observations: list[np.ndarray]):
        """
        The history buffer needs to be reset for new episodes.
        During a new episode, a new (observation,action) pair history will start.
        """
        for buffer, starting_observation in zip(self.buffers, starting_observations):
            buffer.episode_reset(starting_observation=starting_observation)

    def ready(self) -> bool:
        """
        Check to see if the buffer has enough experiences stored to begin learning.
        """
        if self.memory_counter >= self.batch_size:
            return True
        return False

    def sample_buffer(self) -> list[dict]:
        """
        Sample a batch of experiences uniformly from the replay buffer.
        These include (observation,action) pair histories.

        For the multi-agent algorithm,
        it would be convenient to concatenate all (observation,action) pairs for the agents here.

        Returns:
            A list of dictionaries.
            Each element in the list is a dictionary containing the samples for each agent in the MARL algorithm.
            Each dictionary contains a batch of experiences including (observation,action) pair histories.
        """
        # obtain random indices for experiences to sample
        indices = np.random.randint(
            low=self.max_history_length,  # lower bound (inclusive)
            high=self.memory_counter,  # upper bound (exclusive)
            size=self.batch_size  # number of random integers
        )

        # sample batches for each agent
        batches: list[dict] = [buffer.sample_buffer(indices=indices) for buffer in self.buffers]

        # c_s: critic states
        # for the batches, concatenate all agents' (observation,action) pairs
        cat_list: list[torch.Tensor] = [batch['o'] for batch in batches] + [batch['a'] for batch in batches]
        c_s = torch.cat(tensors=cat_list, dim=1)
        for batch in batches:
            batch['c_s'] = c_s

        # h_c_s: history of critic states
        # now I want to concatenate histories of (observation,action) pairs for all agents
        cat_list = [batch['h_o'] for batch in batches] + [batch['h_a'] for batch in batches]
        h_c_s = torch.cat(tensors=cat_list, dim=2)
        for batch in batches:
            batch['h_c_s'] = h_c_s

        # find c_s2 using the history of next observations and actions
        cat_list = [batch['h_o2'] for batch in batches] + [batch['h_a2'] for batch in batches]
        h_c_s2 = torch.cat(tensors=cat_list, dim=2)
        for batch in batches:
            batch['h_c_s2'] = h_c_s2

        return batches

    def store_experience(
            self,
            observations: list[np.ndarray],  # agents could have different size observations
            actions: list[np.ndarray],  # agents could have different size actions
            rewards: list[np.ndarray],  # shape = (number_of_agents,)
            next_observations: list[np.ndarray],  # agents could have different size observations
            terminations: list[np.ndarray]  # shape = (number_of_agents,)
    ):
        """
        Stores information from the environment and agents in the multi-agent replay buffer.

        Takes information as numpy arrays and lists of numpy arrays,
        converts to torch tensors,
        and stores it in the multi-agent replay buffer.

        Args:
            observations (list[np.ndarray]): A list of each agent's environment observations.
            actions (list[np.ndarray]): A list of each agent's actions.
            rewards (list[np.ndarray]): A list of each agent's reward after taking an action.
            next_observations (list[np.ndarray]): A list of each agent's environment observations for the next time step.
            terminations (list[np.ndarray]): A list of boolean flags indicating if each agent has terminated or truncated during the episode.
        """
        for i, buffer in enumerate(self.buffers):
            buffer.store_experience(
                observation=observations[i],
                action=actions[i],
                reward=rewards[i],
                next_observation=next_observations[i],
                termination=terminations[i]
            )

        # increment the memory counter
        self.memory_counter += 1
