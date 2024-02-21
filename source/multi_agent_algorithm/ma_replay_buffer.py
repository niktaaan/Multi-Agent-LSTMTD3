import torch
import numpy as np


class MultiAgentReplayBuffer:

    def __init__(
            self,
            number_of_agents: int,
            buffer_size: int,
            observation_sizes: list[int],
            action_sizes: list[int],
            batch_size: int
    ):
        """
        Args:
            number_of_agents (int): The number of agents (cooperative and competitive) in the multi-agent environment.

            buffer_size (int): The maximum number of experiences/transitions to store in the replay buffer.

            observation_sizes (list[int]): A list of integers representing the observation sizes for each agent in the multi-agent system.

            action_sizes (list[int]): A list of integers representing how many action components (discrete or continuous) each agent's actor network needs to output.

            batch_size (int): The size of each batch for training.

        """
        # save all the passed in argument values
        self.number_of_agents = number_of_agents
        self.buffer_size = buffer_size
        self.observation_sizes = observation_sizes
        self.action_sizes = action_sizes
        self.batch_size = batch_size

        # a counter for keeping track of how many transitions/experiences/memories are stored in the buffer
        self.memory_counter = 0

        # each list element (torch tensors) will store a different agent's experiences
        self.observation_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.next_observation_memory = []
        self.terminal_memory = []

        # add memory storage for each agent to each list
        for i in range(self.number_of_agents):
            # observation memory for each agent
            self.observation_memory.append(
                torch.zeros([self.buffer_size, self.observation_sizes[i]], dtype=torch.float)
            )
            # action memory for each agent
            self.action_memory.append(
                torch.zeros([self.buffer_size, self.action_sizes[i]], dtype=torch.float)
            )
            # reward memory for each agent
            self.reward_memory.append(
                torch.zeros([self.buffer_size], dtype=torch.float)
            )
            # next observation memory for each agent
            self.next_observation_memory.append(
                torch.zeros([self.buffer_size, self.observation_sizes[i]], dtype=torch.float)
            )
            # terminal memory for each agent
            self.terminal_memory.append(
                torch.zeros([self.buffer_size], dtype=torch.bool)
            )

         def store_experience(
            self,
            observations: list[np.ndarray],  # agents could have different size observations
            actions: list[np.ndarray],  # agents could have different size actions
            rewards: list[np.ndarray],  # shape = (number_of_agents,)
            next_observations: list[np.ndarray],  # agents could have different size observations
            terminations: list[np.ndarray]  # shape = (number_of_agents,)
    ):
        """
       

        Args:
            observations (list[np.ndarray]): A list of each agent's environment observations.

            actions (list[np.ndarray]): A list of each agent's actions.

            rewards (list[np.ndarray]): A list of each agent's reward after taking an action.

            next_observations (list[np.ndarray]): A list of each agent's environment observations for the next time step.

            terminations (list[np.ndarray]): A list of boolean flags indicating if each agent has terminated or truncated during the episode.
        """
        # calculate the index in memory to store information
        index = self.memory_counter % self.buffer_size

        # store information
        for agent_index in range(self.number_of_agents):
            # store agent observations
            self.observation_memory[agent_index][index] = torch.tensor(observations[agent_index], dtype=torch.float)

            # store agent actions
            self.action_memory[agent_index][index] = torch.tensor(actions[agent_index], dtype=torch.float)

            # store agent rewards
            self.reward_memory[agent_index][index] = torch.tensor(rewards[agent_index], dtype=torch.float)

            # store agent next observations
            self.next_observation_memory[agent_index][index] = torch.tensor(next_observations[agent_index], dtype=torch.float)

            # store agent terminal flags
            self.terminal_memory[agent_index][index] = torch.tensor(terminations[agent_index], dtype=torch.bool)

        # increment the memory counter
        self.memory_counter += 1
         def sample_buffer(self):
        """
        Uniformly sample a batch of the agents' memories from the buffer for training.
        """
        # before sampling:
        # determine the indices that can be sampled from
        # (don't try to sample from memory indices where memories haven't been stored yet)
        valid_sample_indices = min(self.memory_counter, self.buffer_size)

        # uniformly sample a batch of random indices
        batch = np.random.choice(valid_sample_indices, self.batch_size, replace=False)

        # sample actor information
        observations = []
        actions = []
        rewards = []
        next_observations = []
        terminations = []

        # sample experiences for each agent
        for agent_index in range(self.number_of_agents):
            # sample a batch of observations for each agent
            observations.append(self.observation_memory[agent_index][batch])

            # sample a batch of actions for each agent
            actions.append(self.action_memory[agent_index][batch])

            # sample a batch of rewards for each agent
            rewards.append(self.reward_memory[agent_index][batch])

            # sample a batch of next observations for each agent
            next_observations.append(self.next_observation_memory[agent_index][batch])

            # sample a batch of terminal flags for each agent
            terminations.append(self.terminal_memory[agent_index][batch])

        # return everything as torch tensors that can be easily sent to a GPU
        return observations, actions, rewards, next_observations, terminations

    def ready(self) -> bool:
        """
        Check to see if the buffer has enough experiences stored to begin learning.
        """
        if self.memory_counter >= self.batch_size:
            return True
        return False


if __name__ == '__main__':
    """ Test replay buffer code here. """
    pass
