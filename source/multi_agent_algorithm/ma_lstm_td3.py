import os
import numpy as np
import torch
from source.multi_agent_algorithm.agent_lstm_td3 import LSTMTD3
from source.multi_agent_algorithm.ma_replay_buffer_lstm_td3 import MultiAgentReplayBufferLSTMTD3


class MALSTMTD3:
    def __init__(
            self,
            number_of_agents: int,
            agent_names: list[str],
            observation_sizes: list[int],
            action_sizes: list[int],
            action_space_mins: list[float],
            action_space_maxes: list[float],
            action_noise_std: float = 0.1,
            target_noise: float = 0.2,
            target_noise_clip: float = 0.5,
            actor_learning_rate: float = 1e-4,
            critic_learning_rate: float = 1e-3,
            buffer_size: int = 1_000_000,
            batch_size: int = 128,
            max_history_length: int = 5,
            discount_factor: float = 0.95,
            tau: float = 0.005,
            delay_interval: int = 2,
            critic_mem_pre_lstm_hid_sizes: list[int] = (128,),
            critic_mem_lstm_hid_sizes: list[int] = (128,),
            critic_mem_after_lstm_hid_size: list[int] = (128,),
            critic_cur_feature_hid_sizes: list[int] = (128,),
            critic_post_comb_hid_sizes: list[int] = (128,),
            actor_mem_pre_lstm_hid_sizes: list[int] = (128,),
            actor_mem_lstm_hid_sizes: list[int] = (128,),
            actor_mem_after_lstm_hid_size: list[int] = (128,),
            actor_cur_feature_hid_sizes: list[int] = (128,),
            actor_post_comb_hid_sizes: list[int] = (128,),
            scale_lstm_gradients: bool = False,
            scale_factor_lstm_gradients: float = 2.0
    ):
        # save passed in arguments
        self.number_of_agents: int = number_of_agents
        self.agent_names: list[str] = agent_names
        self.observation_sizes: list[int] = observation_sizes
        self.action_sizes: list[int] = action_sizes
        self.action_space_mins: list[float] = action_space_mins
        self.action_space_maxes: list[float] = action_space_maxes
        self.action_noise_std: float = action_noise_std
        self.target_noise: float = target_noise
        self.target_noise_clip: float = target_noise_clip
        self.actor_learning_rate: float = actor_learning_rate
        self.critic_learning_rate: float = critic_learning_rate
        self.buffer_size: int = buffer_size
        self.batch_size: int = batch_size
        self.max_history_length: int = max_history_length
        self.discount_factor: float = discount_factor
        self.tau: float = tau
        self.delay_interval: int = delay_interval
        self.critic_mem_pre_lstm_hid_sizes: list[int] = critic_mem_pre_lstm_hid_sizes
        self.critic_mem_lstm_hid_sizes: list[int] = critic_mem_lstm_hid_sizes
        self.critic_mem_after_lstm_hid_size: list[int] = critic_mem_after_lstm_hid_size
        self.critic_cur_feature_hid_sizes: list[int] = critic_cur_feature_hid_sizes
        self.critic_post_comb_hid_sizes: list[int] = critic_post_comb_hid_sizes
        self.actor_mem_pre_lstm_hid_sizes: list[int] = actor_mem_pre_lstm_hid_sizes
        self.actor_mem_lstm_hid_sizes: list[int] = actor_mem_lstm_hid_sizes
        self.actor_mem_after_lstm_hid_size: list[int] = actor_mem_after_lstm_hid_size
        self.actor_cur_feature_hid_sizes: list[int] = actor_cur_feature_hid_sizes
        self.actor_post_comb_hid_sizes: list[int] = actor_post_comb_hid_sizes
        self.scale_lstm_gradients: bool = scale_lstm_gradients
        self.scale_factor_lstm_gradients: float = scale_factor_lstm_gradients

        # calculate the critic state size
        self.critic_state_size = sum(observation_sizes) + sum(action_sizes)

        # create the replay buffer (with (observation,action) pair histories) for the agents
        self.ma_replay_buffer = MultiAgentReplayBufferLSTMTD3(
            number_of_agents=self.number_of_agents,
            buffer_size=self.buffer_size,
            observation_sizes=observation_sizes,
            action_sizes=self.action_sizes,
            batch_size=self.batch_size,
            max_history_length=max_history_length
        )

        # create the agents
        self.agents: list[LSTMTD3] = [
            LSTMTD3(
                replay_buffer=self.ma_replay_buffer.buffers[i],
                critic_state_size=self.critic_state_size,
                observation_size=self.observation_sizes[i],
                action_size=self.action_sizes[i],
                action_min=self.action_space_mins[i],
                action_max=self.action_space_maxes[i],
                action_noise_std=self.action_noise_std,
                target_noise=self.target_noise,
                target_noise_clip=self.target_noise_clip,
                actor_learning_rate=self.actor_learning_rate,
                critic_learning_rate=self.critic_learning_rate,
                discount_factor=self.discount_factor,
                tau=self.tau,
                delay_interval=self.delay_interval,
                actor_mem_pre_lstm_hid_sizes=self.actor_mem_pre_lstm_hid_sizes,
                actor_mem_lstm_hid_sizes=self.actor_mem_lstm_hid_sizes,
                actor_mem_after_lstm_hid_size=self.actor_mem_after_lstm_hid_size,
                actor_cur_feature_hid_sizes=self.actor_cur_feature_hid_sizes,
                actor_post_comb_hid_sizes=self.actor_post_comb_hid_sizes,
                critic_mem_pre_lstm_hid_sizes=self.critic_mem_pre_lstm_hid_sizes,
                critic_mem_lstm_hid_sizes=self.critic_mem_lstm_hid_sizes,
                critic_mem_after_lstm_hid_size=self.critic_mem_after_lstm_hid_size,
                critic_cur_feature_hid_sizes=self.critic_cur_feature_hid_sizes,
                critic_post_comb_hid_sizes=self.critic_post_comb_hid_sizes,
                scale_lstm_gradients=self.scale_lstm_gradients,
                scale_factor_lstm_gradients=self.scale_factor_lstm_gradients
            )
            for i in range(self.number_of_agents)
        ]

        # send all the agents to the same device for computations (GPU or CPU)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        for agent in self.agents:
            agent.send_to_device(self.device)

    def episode_reset(self, starting_observations: list[np.ndarray]):
        """
        Each agent's history buffer in the replay buffer needs to be reset.
        During a new episode, a new (observation,action) pair history will start for the agents.
        """
        self.ma_replay_buffer.episode_reset(starting_observations=starting_observations)

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
        Takes information as numpy arrays and lists of numpy arrays.

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

    def choose_action(self, observations: list[np.ndarray], evaluate: bool = False) -> dict[str, np.ndarray]:
        """
        Given the environment observations, each agent chooses an action from its policy.

        Args:
            observations (list[np.ndarray]): A list where each element is a different agent's environment observation.
            evaluate (bool): A flag to turn noise on (evaluate == False) and off (evaluate == True). During training, noise is added to the action for state space exploration. During evaluation, noise is not necessary for testing. (Seeing how the trained agent performs deterministically without noise during evaluation.)

        Returns:
            A dictionary.
            Keys are the agents' IDs.
            Values are np.ndarrays of the agents' actions.
        """
        # each agent's action output from the policies will be stored in a dictionary
        actions = {}

        # get each agents' action and store it in the dictionary
        for name, observation, agent in zip(self.agent_names, observations, self.agents):
            action = agent.choose_action(
                o=torch.tensor(observation, dtype=torch.float).to(self.device),
                evaluate=evaluate
            )
            actions[name] = action

        return actions

    def learn(self):
        # there must be enough experiences stored in the replay buffer for a batch
        if self.ma_replay_buffer.ready() is True:

            # get batches of samples for each agent
            # the batches of samples will be a list of dictionaries,
            # where each element in the list is a dictionary containing the batch of samples for each agent
            batches: list[dict] = self.ma_replay_buffer.sample_buffer()

            # send all pytorch tensors in the batches to the device (GPU)
            for batch in batches:
                for key, tensor in batch.items():
                    batch[key] = tensor.to(self.device)

            # each agent receives its batch of samples and performs its learning step
            for i in range(self.number_of_agents):
                self.agents[i].learn2(
                    agents=self.agents,
                    agent_index=i,
                    batches=batches
                )
