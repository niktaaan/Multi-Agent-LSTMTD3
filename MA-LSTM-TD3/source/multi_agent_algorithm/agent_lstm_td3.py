import os
from copy import deepcopy
import numpy as np
import torch
from source.multi_agent_algorithm.actor_network_lstm_td3 import ActorLSTMTD3
from source.multi_agent_algorithm.critic_network_lstm_td3 import CriticLSTMTD3
from source.multi_agent_algorithm.replay_buffer_lstm_td3 import ReplayBufferLSTMTD3, HistoryBufferLSTMTD3


class LSTMTD3(torch.nn.Module):
    def __init__(
            self,
            replay_buffer: ReplayBufferLSTMTD3,
            critic_state_size: int,
            observation_size: int,
            action_size: int,
            action_min: float,
            action_max: float,
            action_noise_std: float = 0.1,
            target_noise: float = 0.2,
            target_noise_clip: float = 0.5,
            actor_learning_rate: float = 1e-3,
            critic_learning_rate: float = 1e-3,
            discount_factor: float = 0.99,
            tau: float = 0.995,
            delay_interval: int = 2,
            actor_mem_pre_lstm_hid_sizes: list[int] = (128,),
            actor_mem_lstm_hid_sizes: list[int] = (128,),
            actor_mem_after_lstm_hid_size: list[int] = (128,),
            actor_cur_feature_hid_sizes: list[int] = (128,),
            actor_post_comb_hid_sizes: list[int] = (128,),
            critic_mem_pre_lstm_hid_sizes: list[int] = (128,),
            critic_mem_lstm_hid_sizes: list[int] = (128,),
            critic_mem_after_lstm_hid_size: list[int] = (128,),
            critic_cur_feature_hid_sizes: list[int] = (128,),
            critic_post_comb_hid_sizes: list[int] = (128,),
            scale_lstm_gradients: bool = False,
            scale_factor_lstm_gradients: float = 2.0
    ):
        # set up torch.nn.Module
        super(LSTMTD3, self).__init__()

        # keep track of the device
        self.device = 'cpu'

        # the agent needs a reference to its replay buffer and history buffer
        self.replay_buffer = replay_buffer

        # save passed in arguments
        self.critic_state_size = critic_state_size
        self.observation_size = observation_size
        self.action_size = action_size
        self.action_min = action_min
        self.action_max = action_max
        self.action_noise_std = action_noise_std
        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.discount_factor = discount_factor
        self.tau = tau

        # variables for delayed updates
        # timing delayed target network updates
        self.delay_interval = delay_interval
        self.optimization_step_counter = 0

        # save passed in parameters (actor network layer architecture)
        self.actor_mem_pre_lstm_hid_sizes = actor_mem_pre_lstm_hid_sizes
        self.actor_mem_lstm_hid_sizes = actor_mem_lstm_hid_sizes
        self.actor_mem_after_lstm_hid_size = actor_mem_after_lstm_hid_size
        self.actor_cur_feature_hid_sizes = actor_cur_feature_hid_sizes
        self.actor_post_comb_hid_sizes = actor_post_comb_hid_sizes

        # save passed in parameters (critic network layer architecture)
        self.critic_mem_pre_lstm_hid_sizes = critic_mem_pre_lstm_hid_sizes
        self.critic_mem_lstm_hid_sizes = critic_mem_lstm_hid_sizes
        self.critic_mem_after_lstm_hid_size = critic_mem_after_lstm_hid_size
        self.critic_cur_feature_hid_sizes = critic_cur_feature_hid_sizes
        self.critic_post_comb_hid_sizes = critic_post_comb_hid_sizes

        # save passed in parameters (scaling lstm gradients for faster lstm layer training)
        self.scale_lstm_gradients = scale_lstm_gradients
        self.scale_factor_lstm_gradients = scale_factor_lstm_gradients

        # filenames for saving and loading model parameters (TD3)
        self.parameter_filenames = {
            'actor': 'actor.pt',
            'critic_1': 'critic_1.pt',
            'critic_2': 'critic_2.pt',
            'target_actor': 'target_actor.pt',
            'target_critic_1': 'target_critic_1.pt',
            'target_critic_2': 'target_critic_2.pt'
        }

        # create the online networks
        self.actor = ActorLSTMTD3(
            observation_size=self.observation_size,
            action_size=self.action_size,
            action_min=self.action_min,
            action_max=self.action_max,
            mem_pre_lstm_hid_sizes=self.actor_mem_pre_lstm_hid_sizes,
            mem_lstm_hid_sizes=self.actor_mem_lstm_hid_sizes,
            mem_after_lstm_hid_size=self.actor_mem_after_lstm_hid_size,
            cur_feature_hid_sizes=self.actor_cur_feature_hid_sizes,
            post_comb_hid_sizes=self.actor_post_comb_hid_sizes
        )
        self.critic_1 = CriticLSTMTD3(
            critic_state_size=self.critic_state_size,
            mem_pre_lstm_hid_sizes=self.critic_mem_pre_lstm_hid_sizes,
            mem_lstm_hid_sizes=self.critic_mem_lstm_hid_sizes,
            mem_after_lstm_hid_size=self.critic_mem_after_lstm_hid_size,
            cur_feature_hid_sizes=self.critic_cur_feature_hid_sizes,
            post_comb_hid_sizes=self.critic_post_comb_hid_sizes
        )
        self.critic_2 = CriticLSTMTD3(
            critic_state_size=self.critic_state_size,
            mem_pre_lstm_hid_sizes=self.critic_mem_pre_lstm_hid_sizes,
            mem_lstm_hid_sizes=self.critic_mem_lstm_hid_sizes,
            mem_after_lstm_hid_size=self.critic_mem_after_lstm_hid_size,
            cur_feature_hid_sizes=self.critic_cur_feature_hid_sizes,
            post_comb_hid_sizes=self.critic_post_comb_hid_sizes
        )

        # create the target networks
        self.target_actor = deepcopy(self.actor)
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2 = deepcopy(self.critic_2)

        # create the optimizers
        # only the online networks have their parameters optimized
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_1_optimizer = torch.optim.Adam(params=self.critic_1.parameters(), lr=self.critic_learning_rate)
        self.critic_2_optimizer = torch.optim.Adam(params=self.critic_2.parameters(), lr=self.critic_learning_rate)

        # target networks do not have their parameters updated by optimizers
        # they are only updated manually by polyak averaging
        # automatic gradient tracking is not necessary
        for p in self.target_actor.parameters():
            p.requires_grad = False
        for p in self.target_critic_1.parameters():
            p.requires_grad = False
        for p in self.target_critic_2.parameters():
            p.requires_grad = False

    def send_to_device(self, device):
        self.device = device
        self.to(device)

    def save_parameters(self, directory: str):
        """
        Saves all the network parameters to files in the directory.
        """
        # make sure the save directory exists, else create it
        if not os.path.exists(directory):
            os.makedirs(directory)

        # get the state dictionaries
        state_dictionaries = {
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic_1': self.target_critic_1.state_dict(),
            'target_critic_2': self.target_critic_2.state_dict()
        }

        # save (actor, critic_1, critic_2) (TD3)
        torch.save(obj=state_dictionaries['actor'], f=os.path.join(directory, self.parameter_filenames['actor']))
        torch.save(obj=state_dictionaries['critic_1'], f=os.path.join(directory, self.parameter_filenames['critic_1']))
        torch.save(obj=state_dictionaries['critic_2'], f=os.path.join(directory, self.parameter_filenames['critic_2']))

        # save (target_actor, target_critic_1, target_critic_2) (TD3)
        torch.save(obj=state_dictionaries['target_actor'], f=os.path.join(directory, self.parameter_filenames['target_actor']))
        torch.save(obj=state_dictionaries['target_critic_1'], f=os.path.join(directory, self.parameter_filenames['target_critic_1']))
        torch.save(obj=state_dictionaries['target_critic_2'], f=os.path.join(directory, self.parameter_filenames['target_critic_2']))

    def load_parameters(self, directory: str):
        """
        Loads all the network parameters from files in the directory.
        """
        # get the state dictionaries
        state_dictionaries = {
            'actor': torch.load(os.path.join(directory, self.parameter_filenames['actor'])),
            'critic_1': torch.load(os.path.join(directory, self.parameter_filenames['critic_1'])),
            'critic_2': torch.load(os.path.join(directory, self.parameter_filenames['critic_2'])),
            'target_actor': torch.load(os.path.join(directory, self.parameter_filenames['target_actor'])),
            'target_critic_1': torch.load(os.path.join(directory, self.parameter_filenames['target_critic_1'])),
            'target_critic_2': torch.load(os.path.join(directory, self.parameter_filenames['target_critic_2'])),
        }

        # load (actor, critic_1, critic_2) (TD3)
        self.actor.load_state_dict(state_dict=state_dictionaries['actor'])
        self.critic_1.load_state_dict(state_dict=state_dictionaries['critic_1'])
        self.critic_2.load_state_dict(state_dict=state_dictionaries['critic_2'])

        # load (target_actor, target_critic_1, target_critic_2) (TD3)
        self.target_actor.load_state_dict(state_dict=state_dictionaries['target_actor'])
        self.target_critic_1.load_state_dict(state_dict=state_dictionaries['target_critic_1'])
        self.target_critic_2.load_state_dict(state_dict=state_dictionaries['target_critic_2'])

    def choose_action(self, o: torch.cuda.FloatTensor, evaluate: bool = False):
        """
        The agent chooses an action given its current observation from the environment.

        Args:
            o (torch.cuda.FloatTensor): The current observation from the environment. (expected to be on the GPU)
            evaluate: During training, some noise should be added to the agent's action for state space exploration. If evaluate is set to True, then no noise will be added and the actions taken will be deterministic.

        Returns:
            The action bounded/clipped by the action space.
        """
        # get the histories for observations-actions, and their length
        # convert to torch tensors on the device (GPU or CPU) for calculations
        h_b: HistoryBufferLSTMTD3 = self.replay_buffer.history_buffer

        h_o = h_b.observation_buffer \
            .view(1, h_b.observation_buffer.shape[0], h_b.observation_buffer.shape[1]) \
            .float() \
            .to(self.device)

        h_a = h_b.action_buffer \
            .view(1, h_b.action_buffer.shape[0], h_b.action_buffer.shape[1]) \
            .float() \
            .to(self.device)

        h_l = torch.tensor([h_b.observation_buffer_length]) \
            .float() \
            .to(self.device)

        # note:
        # the shapes above needed to be .view(batch_size, buffer.shape[0], buffer.shape[1])
        # because the LSTM unit was expecting batches of histories,
        # but we only have a batch with 1 history to pass through the LSTM to select 1 action

        # choose the action in a forward pass through the actor network
        with torch.no_grad():
            action, _, = self.actor.forward(
                o=o,
                h_o=h_o,
                h_a=h_a,
                h_l=h_l
            )

        # making sure the action is in the proper type, on the right device, and shape to feed to the environment
        action = action.cpu().numpy().reshape(self.action_size)

        # when not evaluating
        # add noise to the deterministic action for exploration of the state space (explore-exploit dilemma)
        if evaluate is False:
            action += self.action_noise_std * np.random.randn(self.action_size)

        # bound/clip the action to the action space and return
        #
        # in the original code it was
        # return np.clip(a, -self.action_max, self.action_max)
        #
        # but this didn't make much sense to me since the bounds on the action would be like
        # [0, 1] == [self.action_min, self.action_max]
        # for most OpenAI environments

        return np.clip(action, self.action_min, self.action_max)

    def update_parameters(self, tau: float):
        """
        Updates the target network weights by a certain amount of the online network weights.

        Args:
            tau (float): tau is the amount of the online weights that will be copied over to the target network, while (1-tau) is the amount of the target weights that will remain the same.
        """
        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            # Note: We use an in-place operations "mul_", "add_" to update target
            # params, as opposed to "mul" and "add", which would make new tensors.
            for p, p_targ in zip(self.actor.parameters(), self.target_actor.parameters()):
                p_targ.data.mul_(1 - tau)
                p_targ.data.add_(tau * p.data)
            for p, p_targ in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
                p_targ.data.mul_(1 - tau)
                p_targ.data.add_(tau * p.data)
            for p, p_targ in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
                p_targ.data.mul_(1 - tau)
                p_targ.data.add_(tau * p.data)

    def calculate_critic_loss(self, batch: dict):
        """
        Computes the losses for the critic networks.

        Args:
            batch (dict): A mini-batch of all the experiences from the replay buffer necessary for learning.

        Returns:
            The critic loss.
        """
        # get the necessary variables from the data dictionary
        r = batch['r']
        o2 = batch['o2']
        d = batch['t']  # d for done flags or terminal flags

        # get the necessary variables from the data dictionary
        h_o2 = batch['h_o2']
        h_a2 = batch['h_a2']
        h_o_l = batch['h_o_l']
        h_o2_l = batch['h_o2_l']

        c_s = batch['c_s']
        h_c_s = batch['h_c_s']
        h_c_s2 = batch['h_c_s2']

        critic_1_values, critic_1_extracted_memory = self.critic_1.forward(c_s=c_s, h_c_s=h_c_s, h_l=h_o_l)
        critic_2_values, critic_2_extracted_memory = self.critic_2.forward(c_s=c_s, h_c_s=h_c_s, h_l=h_o_l)

        # Bellman backup for Q functions
        with torch.no_grad():
            target_actions, _ = self.target_actor.forward(o=o2, h_o=h_o2, h_a=h_a2, h_l=h_o2_l)

            # Target policy smoothing
            # supposedly, this smoothing makes the policy less brittle (won't overfit)
            # https://spinningup.openai.com/en/latest/algorithms/td3.html
            epsilon = torch.randn_like(target_actions) * self.target_noise
            epsilon = torch.clamp(epsilon, -self.target_noise_clip, self.target_noise_clip)
            a2 = target_actions + epsilon
            a2 = torch.clamp(a2, -self.action_max, self.action_max)

            # Using the next actions a2, create the critic state
            c_s2 = torch.cat(tensors=[o2, a2], dim=1)

            # Target Q-values
            target_critic_1_values, _ = self.target_critic_1.forward(c_s=c_s2, h_c_s=h_c_s2, h_l=h_o2_l)
            target_critic_2_values, _ = self.target_critic_2.forward(c_s=c_s2, h_c_s=h_c_s2, h_l=h_o2_l)

            # The TD3 algorithm uses two critics to reduce overestimation bias
            target_critic_min_values = torch.min(target_critic_1_values, target_critic_2_values)
            td_target = r + self.discount_factor * (1 - d) * target_critic_min_values

        # MSE loss against Bellman backup
        critic_1_loss = ((critic_1_values - td_target) ** 2).mean()
        critic_2_loss = ((critic_2_values - td_target) ** 2).mean()

        # a double critic is used for td3 because it gives better value approximations with less estimation bias
        critic_loss = critic_1_loss + critic_2_loss

        # loss info for logging
        # import pdb; pdb.set_trace()
        critic_loss_info = dict(
            Q1Vals=critic_1_values.detach().cpu().numpy(),
            Q2Vals=critic_2_values.detach().cpu().numpy(),
            Q1ExtractedMemory=critic_1_extracted_memory.mean(dim=1).detach().cpu().numpy(),
            Q2ExtractedMemory=critic_2_extracted_memory.mean(dim=1).detach().cpu().numpy()
        )

        return critic_loss, critic_loss_info

    def calculate_actor_loss(self, batch: dict):
        """
        Computes the actor loss (following policy pi)

        Args:
            batch (dict): A mini-batch of all the experiences from the replay buffer necessary for learning.

        Returns:
            Returns the actor loss.
        """
        # get the necessary variables from the data dictionary
        o = batch['o']
        h_o = batch['h_o']
        h_a = batch['h_a']
        h_o_l = batch['h_o_l']

        c_s = batch['c_s']
        h_c_s = batch['h_c_s']

        # get an actions from the actor for the batch
        # calculate the critic_1 q-values for the observation-action pairs and histories in the batch
        actions, actor_extracted_memory = self.actor.forward(o, h_o, h_a, h_o_l)
        critic_1_values, _ = self.critic_1.forward(c_s=c_s, h_c_s=h_c_s, h_l=h_o_l)

        # loss info for logging
        actor_loss_info = dict(
            ActExtractedMemory=actor_extracted_memory.mean(dim=1).detach().cpu().numpy()
        )

        return -critic_1_values.mean(), actor_loss_info

    def learn(self, batch: dict):
        """
        Perform a learning step for the LSTM-TD3 agent.

        Args:
            batch (dict): A mini-batch of all the experiences from the replay buffer necessary for learning.
        """
        # First run one gradient descent step for Q1 and Q2

        # zeroing the critic gradients for a new optimization step
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()

        # compute the critic_loss and perform backpropagation to calculate gradients
        critic_loss, critic_loss_info = self.calculate_critic_loss(batch=batch)
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
            actor_loss, actor_loss_info = self.calculate_actor_loss(batch=batch)
            actor_loss.backward()
            self.actor_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in self.critic_1.parameters():
                p.requires_grad = True
            for p in self.critic_2.parameters():
                p.requires_grad = True

            # (could record the actor loss here and plot later)

            # Finally, update target networks by polyak averaging.
            self.update_parameters(tau=self.tau)

        # an optimization step just finished
        # increment the optimization step counter used for timing target network updates
        self.optimization_step_counter += 1

    def learn2(
            self,
            agents: list,
            agent_index: int,
            batches: list[dict]
    ):
        """
        Perform a learning/optimization step for the current LSTM-TD3 agent in the multi-agent algorithm.

        Args:
            agents: A list of all the reinforcement learning agents in the multi-agent algorithm.
            agent_index (int): The index in the lists of the agent which is currently learning.
            batches (list[dict]): Mini-batches for each agent of all the experiences from the replay buffer necessary for learning.
        """

        """ calculate the target actions with noise, smoothing, and clamping """
        a2 = []
        for agent, batch in zip(agents, batches):
            # agent action without noise, smoothing, clamping
            target_action, _ = agent.target_actor.forward(
                o=batch['o2'],
                h_o=batch['h_o2'],
                h_a=batch['h_a2'],
                h_l=batch['h_o2_l']
            )

            # calculate noise and clamp/clip for adding to action
            noise = torch.randn_like(target_action) * self.target_noise
            noise = torch.clamp(
                input=noise,
                min=-self.target_noise_clip,
                max=self.target_noise_clip
            )

            # add noise to action
            target_action = target_action + noise

            # clamp/clip action to action space bounds
            target_action = torch.clamp(
                input=target_action,
                min=self.action_min,
                max=self.action_max
            )

            # append the agent's target action to the list of target actions
            a2.append(target_action)

        """ get the critic values for the target next critic states """
        c_s2 = torch.cat(
            [
                torch.cat([batch['o2'] for batch in batches], dim=1),
                torch.cat(a2, dim=1)
            ],
            dim=1
        )

        # calculate the batch of q-values using the two target critics (for the next critic states)
        q1_ = self.target_critic_1.forward(c_s=c_s2, h_c_s=batches[agent_index]['h_c_s2'], h_l=batches[agent_index]['h_o2_l'])[0].squeeze()
        q2_ = self.target_critic_2.forward(c_s=c_s2, h_c_s=batches[agent_index]['h_c_s2'], h_l=batches[agent_index]['h_o2_l'])[0].squeeze()

        # calculate the batch of q-values using the two online critics (for the current critic states)
        q1 = self.critic_1.forward(c_s=batches[agent_index]['c_s'], h_c_s=batches[agent_index]['h_c_s'], h_l=batches[agent_index]['h_o_l'])[0].squeeze()
        q2 = self.critic_2.forward(c_s=batches[agent_index]['c_s'], h_c_s=batches[agent_index]['h_c_s'], h_l=batches[agent_index]['h_o_l'])[0].squeeze()

        # make use of the done mask to set terminal states to have values of 0.0
        # here is an example of how this masking works
        # bool_list = torch.tensor([False, True, False], dtype=torch.bool)
        # my_tensor = torch.tensor([10.0, 10.0, 10.0])
        # my_tensor[bool_list] = 0.0
        # print(my_tensor) # this gives [10.0, 0.0, 10.0]
        q1_[batches[agent_index]['t'].bool()] = 0.0
        q2_[batches[agent_index]['t'].bool()] = 0.0

        # reshape the target critic values
        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        # take only the minimum values of the target critic values
        q_ = torch.min(q1_, q2_)

        # calculate the TD target values (these are the predicted values/returns of the next states)
        target = batches[agent_index]['r'] + self.discount_factor * q_
        target = target.view(-1)  # reshape

        # zero gradients before performing backpropagation for the critic networks
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()

        # calculate the critic networks' losses
        q1_loss = torch.nn.functional.mse_loss(target, q1)
        q2_loss = torch.nn.functional.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss

        # perform the backpropagation step to calculate gradients
        critic_loss.backward()

        # scaling the lstm gradients before optimization step for faster training
        # (this might work, or it might make the optimization unstable like when the learning rate is too high)
        if self.scale_lstm_gradients is True:
            for layer_1, layer_2 in zip(self.critic_1.mem_lstm_layers, self.critic_2.mem_lstm_layers):
                for p1, p2 in zip(layer_1.parameters(), layer_2.parameters()):
                    p1.grad *= self.scale_factor_lstm_gradients
                    p2.grad *= self.scale_factor_lstm_gradients

        # perform optimization step using calculated gradients
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        """ TD3 delayed update of the actor and target networks """
        if self.optimization_step_counter % self.delay_interval == 0:
            # zero the actor network gradients
            self.actor_optimizer.zero_grad()

            # all agents' actions are calculated for the batch
            a = [
                agent.actor.forward(
                    o=batch['o'],
                    h_o=batch['h_o'],
                    h_a=batch['h_a'],
                    h_l=batch['h_o_l']
                )[0]
                for agent, batch in zip(agents, batches)
            ]

            # get the critic state (all agents' actions and observations concatenated
            c_s = torch.cat(
                [
                    torch.cat([batch['o'] for batch in batches], dim=1),
                    torch.cat(a, dim=1)
                ],
                dim=1
            )

            # calculate the actor loss
            actor_q1_loss = self.critic_1.forward(
                c_s=c_s,
                h_c_s=batches[agent_index]['h_c_s'],
                h_l=batches[agent_index]['h_o_l']
            )[0].squeeze()
            actor_loss = -torch.mean(actor_q1_loss)

            # perform backpropagation and calculate the gradients for the actor network
            actor_loss.backward()

            # scaling the lstm gradients before optimization step for faster training
            # (this might work, or it might make the optimization unstable like when the learning rate is too high)
            if self.scale_lstm_gradients is True:
                for layer in self.actor.mem_lstm_layers:
                    for param in layer.parameters():
                        param.grad *= self.scale_factor_lstm_gradients

            # use the gradients and perform an optimization step for the actor network
            self.actor_optimizer.step()

            # update the target network parameters
            self.update_parameters(tau=self.tau)

        # an optimization step just finished
        # increment the optimization step counter used for timing target network updates
        self.optimization_step_counter += 1
