"""
Description:
    This is a version of the MADDPG algorithm where agents other than DDPG can be used.
    This class is a wrapper class that coordinates the different agents in the Multi-Agent Algorithm.

Author:
    Jordan Cramer

Date:
    2023-08-23
"""
import os
import torch
import numpy as np
from copy import deepcopy
from source.multi_agent_algorithm.agent_ddpg import DDPG
from source.multi_agent_algorithm.ma_replay_buffer import MultiAgentReplayBuffer


class MADDPG:
    """
    A wrapper class that coordinates different agents in the Multi-agent Algorithm.
    This is a version of the MADDPG algorithm.

    This class should coordinate,

    1. All agents
    2. Agent actions
    3. Experience replay buffer
    4. Saving/Loading agent parameters and replay buffer states from checkpoint files
    """
    def __init__(
            self,
            number_of_agents: int,
            agent_names: list[str],
            observation_sizes: list[int],
            action_sizes: list[int],
            action_space_mins: list[float],
            action_space_maxes: list[float],
            action_noise_std: float = 0.1,
            actor_learning_rate: float = 1e-4,
            critic_learning_rate: float = 1e-3,
            actor_layer_1_size: int = 64,
            actor_layer_2_size: int = 64,
            critic_layer_1_size: int = 64,
            critic_layer_2_size: int = 64,
            buffer_size: int = 1_000_000,
            batch_size: int = 128,
            discount_factor: float = 0.95,
            tau: float = 0.01
    ):
        """
        Args:
            number_of_agents (int): The number of agents in the environment/algorithm.
            agent_names (list[str]): A list of the agent names in the environment. From PettingZoo environments: agent_names = env.agents Example: ['adversary_0', 'agent_0', 'agent_1']
            observation_sizes (list[int]): A list of the agents' observation sizes.
            action_sizes (list[int]): A list of the number of action components to output from each agent's actor networks.
            action_space_mins (list[float]): A list containing the lower bounds (minimums) for values for each agent's actions.
            action_space_maxes (list[float]): A list containing the upper bounds (maximums) for values for each agent's actions.
            action_noise_std (float): For DDPG agents, noise is added to their actions. This action_noise parameter value is the standard deviation for a normal distribution, mean=0, that the noise is sampled from.
            actor_learning_rate (float): The learning rate of the actor network.
            critic_learning_rate (float): The learning rate of the critic network.
            actor_layer_1_size (int): The output size of the first fully connected linear layer.
            actor_layer_2_size (int): The output size of the second fully connected linear layer.
            critic_layer_1_size (int): The output size of the first fully connected linear layer.
            critic_layer_2_size (int): The output size of the second fully connected linear layer.
            buffer_size (int): The maximum number of experiences/transitions to store in the replay buffer.
            batch_size (int): The size of each batch for training.
            discount_factor (float): The discount factor for discounted returns.
            tau (float): tau is the amount of the online weights that will be copied over to the target network, while (1-tau) is the amount of the target weights that will remain the same.
        """
        # save passed in arguments
        self.number_of_agents = number_of_agents
        self.agent_names = deepcopy(agent_names)
        self.observation_sizes = observation_sizes
        self.action_sizes = action_sizes
        self.action_space_mins = action_space_mins
        self.action_space_maxes = action_space_maxes
        self.action_noise_std = action_noise_std
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.actor_layer_1_size = actor_layer_1_size
        self.actor_layer_2_size = actor_layer_2_size
        self.critic_layer_1_size = critic_layer_1_size
        self.critic_layer_2_size = critic_layer_2_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.tau = tau

        # calculate the critic state size (all agents' observations and actions concatenated)
        self.critic_state_size = sum(observation_sizes) + sum(action_sizes)

        # a list to hold the agents
        self.agents: list[DDPG] = []

        # a replay buffer for the multi-agent algorithm
        self.ma_replay_buffer = MultiAgentReplayBuffer(
            number_of_agents=number_of_agents,
            buffer_size=buffer_size,
            observation_sizes=observation_sizes,
            action_sizes=action_sizes,
            batch_size=batch_size
        )

        # create each agent and append to the list of agents
        for agent_index in range(number_of_agents):
            self.agents.append(
                DDPG(
                    name=agent_names[agent_index],
                    observation_size=observation_sizes[agent_index],
                    critic_state_size=self.critic_state_size,
                    action_size=action_sizes[agent_index],
                    action_min=action_space_mins[agent_index],
                    action_max=action_space_maxes[agent_index],
                    action_noise_std=action_noise_std,
                    actor_learning_rate=actor_learning_rate,
                    critic_learning_rate=critic_learning_rate,
                    actor_layer_1_size=actor_layer_1_size,
                    actor_layer_2_size=actor_layer_2_size,
                    critic_layer_1_size=critic_layer_1_size,
                    critic_layer_2_size=critic_layer_2_size,
                    discount_factor=discount_factor,
                    tau=tau
                )
            )

    def save_checkpoint(self, directory: str, save_replay_buffer: bool = True):
        """ Saves each agent's neural network weights to the directory. Saves the replay buffer. """
        # create the directories and save the parameters
        for index, name in enumerate(self.agent_names):

            # the directory should exist first
            if not os.path.exists(os.path.join(directory, 'checkpoint', name)):
                os.makedirs(os.path.join(directory, 'checkpoint', name))

            # save the agent parameters
            self.agents[index].save_parameters(directory=os.path.join(directory, 'checkpoint', name))

        # save the replay buffer
        if save_replay_buffer is True:
            torch.save(obj=self.ma_replay_buffer, f=os.path.join(directory, 'checkpoint', 'replay_buffer.pt'))

    def load_checkpoint(self, directory: str, load_replay_buffer: bool = True):
        """ Loads each agent's neural network weights from the directory. Loads the saved replay buffer. """
        # load the agent parameters
        directory = os.path.join(directory, 'checkpoint')
        for index, name in enumerate(self.agent_names):
            self.agents[index].load_parameters(directory=os.path.join(directory, name))

        # load the replay buffer
        if load_replay_buffer is True:
            self.ma_replay_buffer = torch.load(f=os.path.join(directory, 'replay_buffer.pt'))

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
        self.ma_replay_buffer.store_experience(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminations=terminations
        )

    def choose_action(self, observations: list[torch.Tensor], evaluate: bool = False) -> dict[str, np.ndarray]:
        """
        Given the environment observations, each agent chooses an action from its policy.

        Args:
            observations (list[torch.Tensor]): A list where each element is a different agent's environment observation.

            evaluate (bool): A flag to turn noise on and off. During training, noise is added to the action for state space exploration. During evaluation, noise is not necessary for testing. (Seeing how the trained agent performs deterministically without noise during evaluation.)

        Returns:
            A dictionary.
            Keys are the agents' IDs.
            Values are np.ndarrays of the agents' actions.
        """
        # each agent's action output from the policies will be stored in a dictionary
        actions = {}

        # get each agents' action and store it in the dictionary
        for agent_name, observation, agent in zip(self.agent_names, observations, self.agents):
            action = agent.choose_action(observation, evaluate)
            actions[agent_name] = action

        return actions

    def learn(self):
        """
        Each agent will perform an optimization/update/learning step.
        """
        # make sure there are enough experiences stored in the replay buffer before learning
        # there needs to be at least enough experiences for a batch before learning
        if self.ma_replay_buffer.ready():
            # each agent will sample experiences from the buffer and learn
            for index, agent in enumerate(self.agents):
                observations, actions, rewards, next_observations, terminations = self.ma_replay_buffer.sample_buffer()
                agent.learn(
                    agent_list=self.agents,
                    agent_index=index,
                    observations=observations,
                    actions=actions,
                    rewards=rewards,
                    next_observations=next_observations,
                    terminations=terminations
                )
