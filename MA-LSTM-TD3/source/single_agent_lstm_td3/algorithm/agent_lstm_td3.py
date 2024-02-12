"""
Description:
    The agent for a LSTM-TD3 reinforcement learning agent.

    I mostly added comments and refactored a little for readability,
    and to make it consistent with other coded algorithms for the multi-agent program.

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
from copy import deepcopy
import numpy as np
import torch
from source.single_agent_lstm_td3.algorithm.actor_network_lstm_td3 import ActorNetwork
from source.single_agent_lstm_td3.algorithm.critic_network_lstm_td3 import CriticNetwork
from source.single_agent_lstm_td3.algorithm.replay_buffer_lstm_td3 import ReplayBuffer
from source.single_agent_lstm_td3.algorithm.history_buffer_lstm_td3 import HistoryBuffer


class LSTMTD3(torch.nn.Module):
    def __init__(
            self,
            observation_shape,
            action_shape,
            action_min: float,
            action_max: float,
            action_noise_std: float = 0.1,
            target_noise: float = 0.2,
            target_noise_clip: float = 0.5,
            buffer_size: int = int(1e6),
            max_history_length: int = 5,
            discount_factor: float = 0.99,
            tau: float = 0.995,
            actor_learning_rate: float = 1e-3,
            critic_learning_rate: float = 1e-3,
            delay_interval: int = 2,
            batch_size: int = 100,
            critic_hist_with_past_act: bool = False,
            actor_hist_with_past_act: bool = False,
            use_double_critic: bool = True,
            use_target_policy_smoothing: bool = True,
            critic_mem_pre_lstm_hid_sizes: tuple = (128,),
            critic_mem_lstm_hid_sizes: tuple = (128,),
            critic_mem_after_lstm_hid_size: tuple = (128,),
            critic_cur_feature_hid_sizes: tuple = (128,),
            critic_post_comb_hid_sizes: tuple = (128,),
            actor_mem_pre_lstm_hid_sizes: tuple = (128,),
            actor_mem_lstm_hid_sizes: tuple = (128,),
            actor_mem_after_lstm_hid_size: tuple = (128,),
            actor_cur_feature_hid_sizes: tuple = (128,),
            actor_post_comb_hid_sizes: tuple = (128,),
    ):
        """
        Args:
            observation_shape (int):

            action_shape (int):

            action_min (float):

            action_max (float):

            action_noise_std:

            target_noise:

            target_noise_clip:

            buffer_size (int):

            max_history_length (int):

            discount_factor:

            tau:

            actor_learning_rate:

            critic_learning_rate:

            delay_interval (int):

            batch_size (int):

            critic_hist_with_past_act:

            actor_hist_with_past_act:

            use_double_critic: Yes, two critic networks to reduce overestimation bias will always be used with the TD3 algorithm.

            use_target_policy_smoothing:

            critic_mem_pre_lstm_hid_sizes (int):

            critic_mem_lstm_hid_sizes (int):

            critic_mem_after_lstm_hid_size (int):

            critic_cur_feature_hid_sizes (int):

            critic_post_comb_hid_sizes (int):

            actor_mem_pre_lstm_hid_sizes (int):

            actor_mem_lstm_hid_sizes (int):

            actor_mem_after_lstm_hid_size (int):

            actor_cur_feature_hid_sizes (int):

            actor_post_comb_hid_sizes (int):
        """

        # run the constructor for torch.nn.Module for its functionality
        super(LSTMTD3, self).__init__()

        """ simply store the passed in argument values """

        # basic arguments
        self.observation_shape = observation_shape
        self.action_shape = action_shape

        # noise, clipping, and action value bounds
        self.action_min = action_min
        self.action_max = action_max
        self.action_noise_std = action_noise_std
        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip

        # replay buffer size and lstm observation-action history length
        self.buffer_size = buffer_size
        self.max_history_length = max_history_length

        # discount factor, learning rates, polyak averaging (target network weight updates)
        self.discount_factor = discount_factor
        self.tau = tau
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.batch_size = batch_size

        # flags
        self.critic_hist_with_past_act = critic_hist_with_past_act
        self.actor_hist_with_past_act = actor_hist_with_past_act
        self.use_double_critic = use_double_critic
        self.use_target_policy_smoothing = use_target_policy_smoothing

        # critic layer sizes
        self.critic_mem_pre_lstm_hid_sizes = critic_mem_pre_lstm_hid_sizes
        self.critic_mem_lstm_hid_sizes = critic_mem_lstm_hid_sizes
        self.critic_mem_after_lstm_hid_size = critic_mem_after_lstm_hid_size
        self.critic_cur_feature_hid_sizes = critic_cur_feature_hid_sizes
        self.critic_post_comb_hid_sizes = critic_post_comb_hid_sizes

        # actor layer sizes
        self.actor_mem_pre_lstm_hid_sizes = actor_mem_pre_lstm_hid_sizes
        self.actor_mem_lstm_hid_sizes = actor_mem_lstm_hid_sizes
        self.actor_mem_after_lstm_hid_size = actor_mem_after_lstm_hid_size
        self.actor_cur_feature_hid_sizes = actor_cur_feature_hid_sizes
        self.actor_post_comb_hid_sizes = actor_post_comb_hid_sizes

        # variables for delayed updates
        # timing delayed target network updates
        self.delay_interval = delay_interval
        self.optimization_step_counter = 0

        # replay buffer for storing agent experiences
        self.replay_buffer = ReplayBuffer(observation_shape=observation_shape, action_shape=action_shape, buffer_size=buffer_size)

        # history buffer for storing agent observation-action histories
        self.history_buffer = HistoryBuffer(observation_shape=observation_shape, action_shape=action_shape, max_history_length=max_history_length)

        # create the online actor network
        self.actor = ActorNetwork(
            observation_shape=observation_shape,
            action_shape=action_shape,
            action_min=action_min,
            action_max=action_max,
            mem_pre_lstm_hid_sizes=actor_mem_pre_lstm_hid_sizes,
            mem_lstm_hid_sizes=actor_mem_lstm_hid_sizes,
            mem_after_lstm_hid_size=actor_mem_after_lstm_hid_size,
            cur_feature_hid_sizes=actor_cur_feature_hid_sizes,
            post_comb_hid_sizes=actor_post_comb_hid_sizes,
            hist_with_past_act=actor_hist_with_past_act
        )

        # create the online critic 1 network
        self.critic_1 = CriticNetwork(
            observation_shape=observation_shape,
            action_shape=action_shape,
            mem_pre_lstm_hid_sizes=critic_mem_pre_lstm_hid_sizes,
            mem_lstm_hid_sizes=critic_mem_lstm_hid_sizes,
            mem_after_lstm_hid_size=critic_mem_after_lstm_hid_size,
            cur_feature_hid_sizes=critic_cur_feature_hid_sizes,
            post_comb_hid_sizes=critic_post_comb_hid_sizes,
            hist_with_past_act=critic_hist_with_past_act
        )

        # create the online critic 2 network
        self.critic_2 = CriticNetwork(
            observation_shape=observation_shape,
            action_shape=action_shape,
            mem_pre_lstm_hid_sizes=critic_mem_pre_lstm_hid_sizes,
            mem_lstm_hid_sizes=critic_mem_lstm_hid_sizes,
            mem_after_lstm_hid_size=critic_mem_after_lstm_hid_size,
            cur_feature_hid_sizes=critic_cur_feature_hid_sizes,
            post_comb_hid_sizes=critic_post_comb_hid_sizes,
            hist_with_past_act=critic_hist_with_past_act
        )

        # create the target networks
        # these networks will also have the same initial weights as the online networks after deepcopying
        self.target_actor = deepcopy(self.actor)
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2 = deepcopy(self.critic_2)

        # target networks do not have their parameters updated by optimizers
        # they are only updated manually by polyak averaging
        # automatic gradient tracking is not necessary
        for p in self.target_actor.parameters():
            p.requires_grad = False
        for p in self.target_critic_1.parameters():
            p.requires_grad = False
        for p in self.target_critic_2.parameters():
            p.requires_grad = False

        # set up optimizers for the actor and critic networks
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_learning_rate)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_learning_rate)

        # send the agent and networks to the GPU for faster calculations if possible, else use the CPU for calculations
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def initialize_history_buffer(self, starting_observation):
        """
        At the beginning of each new episode, the history buffer should be initialized.

        Args:
            starting_observation: The starting observation from the environment after the environment has been reset.
        """
        self.history_buffer.initialize(
            starting_observation=starting_observation
        )

    def add_observation_action_to_history_buffer(self, observation, action):
        """
        Every time there is a new observation, add it to the history buffer.

        Args:
            observation: The observation from the environment to be added to the history buffer.

            action: The action of the agent to be added to the history buffer.
        """
        self.history_buffer.add_observation_action(
            observation=observation,
            action=action
        )

    def act(self, obs, hist_obs=None, hist_act=None, hist_seg_len=None):
        """
        Returns an unclipped and unscaled agent action given the environment observation.
        The action

        Args:
            obs: The current observation from the environment.

            hist_obs: The previous environment observations from previous time steps.

            hist_act: The previous actions the agent took.

            hist_seg_len: The length/number of previous observations and actions in the buffers.

        Returns:
            Returns an unclipped and unscaled agent action given the environment observation.
        """
        if (hist_obs is None) or (hist_act is None) or (hist_seg_len is None):
            hist_obs = torch.zeros(1, 1, self.observation_shape).to(self.device)
            hist_act = torch.zeros(1, 1, self.action_shape).to(self.device)
            hist_seg_len = torch.zeros(1).to(self.device)

        with torch.no_grad():
            action, _, = self.actor.forward(obs, hist_obs, hist_act, hist_seg_len)

            return action.cpu().numpy()

    def choose_action(self, observation, add_noise: bool = True):
        """
        Returns a clipped action that had noise for exploration added to it.
        This is the function that will be primarily used by the agent to choose actions to take in the environment.

        Args:
            observation: The current observation from the environment.

            add_noise (bool): During training, some noise should be added to the agent's action for state space exploration.

        Returns:
            The clipped action that had noise for state space exploration added to it.
        """
        # get the histories for observations-actions, and their length
        # convert to torch tensors on the device (GPU or CPU) for calculations
        h_o = torch.tensor(self.history_buffer.observation_buffer)\
            .view(1, self.history_buffer.observation_buffer.shape[0], self.history_buffer.observation_buffer.shape[1])\
            .float()\
            .to(self.device)

        h_a = torch.tensor(self.history_buffer.action_buffer)\
            .view(1, self.history_buffer.action_buffer.shape[0], self.history_buffer.action_buffer.shape[1])\
            .float()\
            .to(self.device)

        h_l = torch.tensor([self.history_buffer.observation_buffer_length])\
            .float()\
            .to(self.device)

        # choose the action without noise or clipping
        # it is not necessary to waste computational effort when the gradient tracking isn't necessary
        with torch.no_grad():
            action = self.act(
                torch.as_tensor(observation, dtype=torch.float32).view(1, -1).to(self.device),
                h_o,
                h_a,
                h_l
            ).reshape(self.action_shape)

        # add noise to the deterministic action for exploration of the state space (explore-exploit dilemma)
        if add_noise:
            action += self.action_noise_std * np.random.randn(self.action_shape)

        # in the original code it was
        # return np.clip(a, -self.action_max, self.action_max)
        #
        # but this didn't make much sense to me since the bounds on the action would be like
        # [0, 1] == [self.action_min, self.action_max]
        # for most OpenAI environments

        return np.clip(action, self.action_min, self.action_max)

    def calculate_critic_loss(self, data: dict):
        """
        Computes the losses for the critic networks.

        Args:
            data (dict):

        Returns:
            The critic losses.
        """
        # get the necessary variables from the data dictionary
        observations = data['observations']
        actions = data['actions']
        r = data['rewards']
        o2 = data['next_observations']
        d = data['terminations']  # d for done flags or terminal flags

        # get the necessary variables from the data dictionary
        h_o = data['history_observations']
        h_a = data['history_actions']
        h_o2 = data['history_next_observations']
        h_a2 = data['history_next_actions']
        h_o_len = data['history_observations_length']
        h_o2_len = data['history_next_observations_length']

        critic_1_values, critic_1_extracted_memory = self.critic_1.forward(observations, actions, h_o, h_a, h_o_len)
        critic_2_values, critic_2_extracted_memory = self.critic_2.forward(observations, actions, h_o, h_a, h_o_len)

        # Bellman backup for Q functions
        with torch.no_grad():
            target_actions, _ = self.target_actor.forward(o2, h_o2, h_a2, h_o2_len)

            # Target policy smoothing
            # supposedly, this smoothing makes the policy less brittle (won't overfit)
            # https://spinningup.openai.com/en/latest/algorithms/td3.html
            if self.use_target_policy_smoothing:
                epsilon = torch.randn_like(target_actions) * self.target_noise
                epsilon = torch.clamp(epsilon, -self.target_noise_clip, self.target_noise_clip)
                a2 = target_actions + epsilon
                a2 = torch.clamp(a2, -self.action_max, self.action_max)
            else:
                a2 = target_actions

            # Target Q-values
            target_critic_1_values, _ = self.target_critic_1.forward(o2, a2, h_o2, h_a2, h_o2_len)
            target_critic_2_values, _ = self.target_critic_2.forward(o2, a2, h_o2, h_a2, h_o2_len)

            # The TD3 algorithm uses two critics to reduce overestimation bias
            if self.use_double_critic:
                target_critic_min_values = torch.min(target_critic_1_values, target_critic_2_values)
            else:
                target_critic_min_values = target_critic_1_values
            td_target = r + self.discount_factor * (1 - d) * target_critic_min_values

        # MSE loss against Bellman backup
        critic_1_loss = ((critic_1_values - td_target) ** 2).mean()
        critic_2_loss = ((critic_2_values - td_target) ** 2).mean()

        if self.use_double_critic:
            critic_loss = critic_1_loss + critic_2_loss
        else:
            critic_loss = critic_1_loss

        # loss info for logging
        # import pdb; pdb.set_trace()
        critic_loss_info = dict(
            Q1Vals=critic_1_values.detach().cpu().numpy(),
            Q2Vals=critic_2_values.detach().cpu().numpy(),
            Q1ExtractedMemory=critic_1_extracted_memory.mean(dim=1).detach().cpu().numpy(),
            Q2ExtractedMemory=critic_2_extracted_memory.mean(dim=1).detach().cpu().numpy()
        )

        return critic_loss, critic_loss_info

    def calculate_actor_loss(self, data: dict):
        """
        Computes the actor loss (following policy pi)

        Args:
            data (dict):

        Returns:
            Returns the actor loss.
        """
        # get the necessary variables from the data dictionary
        observations = data['observations']
        h_o = data['history_observations']
        h_a = data['history_actions']
        h_o_len = data['history_observations_length']

        # get an actions from the actor for the batch
        # calculate the critic_1 q-values for the observation-action pairs and histories in the batch
        actions, actor_extracted_memory = self.actor.forward(observations, h_o, h_a, h_o_len)
        critic_1_values, _ = self.critic_1.forward(observations, actions, h_o, h_a, h_o_len)

        # loss info for logging
        actor_loss_info = dict(
            ActExtractedMemory=actor_extracted_memory.mean(dim=1).detach().cpu().numpy()
        )

        return -critic_1_values.mean(), actor_loss_info

    def learn(self, data: dict):
        """
        Args:
            data (dict):
        """
        # First run one gradient descent step for Q1 and Q2

        # zeroing the critic gradients for a new optimization step
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()

        # compute the critic_loss and perform backpropagation to calculate gradients
        critic_loss, critic_loss_info = self.calculate_critic_loss(data=data)
        critic_loss.backward()

        # optimizing the critic networks using the calculated gradients during backpropagation
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # could record the critic loss here and plot later

        # TD3 delayed update of the actor and target networks
        if self.optimization_step_counter % self.delay_interval == 0:
            # Freeze critic networks,
            # so computational effort is not wasted computing gradients for them during the policy learning step.
            for p in self.critic_1.parameters():
                p.requires_grad = False
            for p in self.critic_2.parameters():
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            self.actor_optimizer.zero_grad()
            actor_loss, actor_loss_info = self.calculate_actor_loss(data=data)
            actor_loss.backward()
            self.actor_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in self.critic_1.parameters():
                p.requires_grad = True
            for p in self.critic_2.parameters():
                p.requires_grad = True

            # (could record the actor loss here and plot later)

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                # Note: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                for p, p_targ in zip(self.actor.parameters(), self.target_actor.parameters()):
                    p_targ.data.mul_(self.tau)
                    p_targ.data.add_((1 - self.tau) * p.data)
                for p, p_targ in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
                    p_targ.data.mul_(self.tau)
                    p_targ.data.add_((1 - self.tau) * p.data)
                for p, p_targ in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
                    p_targ.data.mul_(self.tau)
                    p_targ.data.add_((1 - self.tau) * p.data)

        # an optimization step just finished
        # increment the optimization step counter used for timing target network updates
        self.optimization_step_counter += 1
