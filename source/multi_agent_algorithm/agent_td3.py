
import os
from copy import deepcopy
import numpy as np
import torch
from source.multi_agent_algorithm.critic_network import CriticNetwork
from source.multi_agent_algorithm.actor_network import ActorNetwork


class TD3:

    def __init__(
            self,
            name: str,
            observation_size: int,
            critic_state_size: int,
            action_size: int,
            action_min: float,
            action_max: float,
            action_noise_std: float = 0.1,
            actor_learning_rate: float = 1e-4,
            critic_learning_rate: float = 1e-3,
            actor_layer_1_size: int = 64,
            actor_layer_2_size: int = 64,
            critic_layer_1_size: int = 64,
            critic_layer_2_size: int = 64,
            discount_factor: float = 0.95,
            tau: float = 0.01,
            delay_interval: int = 2  # every X number of time steps, update the actor and critic network weights (delayed part)
    ):
        """
        Args:
            name (str): A name given to the agent.
            observation_size (int): The observation size to the actor network. The size is just the size of the individual agent's observation.
            critic_state_size (int): The state size to the critic network. The state size is the size of all agents states and actions combined (For the MADDPG algorithm).
            action_size (int): The number of action components to output from the actor network.
            action_min (float): The minimum value/bound for an action component.
            action_max (float): The maximum value/bound for an action component.
            action_noise_std (float): For TD3 agents, noise is added to their actions. This action_noise parameter value is the standard deviation for a normal distribution, mean=0, that the noise is sampled from.
            actor_learning_rate (float): The learning rate of the actor network.
            critic_learning_rate (float): The learning rate of the critic network.
            actor_layer_1_size (int): The output size of the first fully connected linear layer.
            actor_layer_2_size (int): The output size of the second fully connected linear layer.
            critic_layer_1_size (int): The output size of the first fully connected linear layer.
            critic_layer_2_size (int): The output size of the second fully connected linear layer.
            discount_factor (float): The discount factor for discounted returns.
            tau (float): tau is the amount of the online weights that will be copied over to the target network, while (1-tau) is the amount of the target weights that will remain the same.
            delay_interval (int): The target actor and critic network weights are updated after this number of learning/optimization steps.
        """
        # save all the passed in arguments
        self.name = name
        self.observation_size = observation_size
        self.critic_state_size = critic_state_size
        self.action_size = action_size
        self.action_min = action_min
        self.action_max = action_max
        self.action_noise_std = action_noise_std
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.actor_layer_1_size = actor_layer_1_size
        self.actor_layer_2_size = actor_layer_2_size
        self.critic_layer_1_size = critic_layer_1_size
        self.critic_layer_2_size = critic_layer_2_size
        self.discount_factor = discount_factor
        self.tau = tau

        # variables for delayed updates
        # timing delayed target network updates
        self.delay_interval = delay_interval
        self.optimization_step_counter = 0

        # filenames for saving and loading model parameters (TD3)
        self.parameter_filenames = {
            'actor': 'actor.pt',
            'critic_1': 'critic_1.pt',
            'critic_2': 'critic_2.pt',
            'target_actor': 'target_actor.pt',
            'target_critic_1': 'target_critic_1.pt',
            'target_critic_2': 'target_critic_2.pt'
        }

        # create the online actor, critic_1, and critic_2 networks
        self.actor = ActorNetwork(
            learning_rate=actor_learning_rate,
            action_size=action_size,
            action_min=action_min,
            action_max=action_max,
            input_size=observation_size,
            layer_1_size=actor_layer_1_size,
            layer_2_size=actor_layer_2_size
        )
        self.critic_1 = CriticNetwork(
            learning_rate=critic_learning_rate,
            input_size=critic_state_size,
            layer_1_size=critic_layer_1_size,
            layer_2_size=critic_layer_2_size
        )
        self.critic_2 = CriticNetwork(
            learning_rate=critic_learning_rate,
            input_size=critic_state_size,
            layer_1_size=critic_layer_1_size,
            layer_2_size=critic_layer_2_size
        )

        # create the target actor and critic networks
        # note:
        # deepcopy() will also ensure that the weights are the same for the target networks
        self.target_actor = deepcopy(self.actor)
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2 = deepcopy(self.critic_2)

        # send the networks to GPUs if available, else CPUs
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.actor.send_to_device(self.device)
        self.critic_1.send_to_device(self.device)
        self.critic_2.send_to_device(self.device)
        self.target_actor.send_to_device(self.device)
        self.target_critic_1.send_to_device(self.device)
        self.target_critic_2.send_to_device(self.device)

    def update_parameters(self, tau: float):
        """
        Updates the target network weights by a certain amount of the online network weights.

        Args:
            tau (float): tau is the amount of the online weights that will be copied over to the target network, while (1-tau) is the amount of the target weights that will remain the same.
        """
        # update target actor weights
        for actor_parameter, target_actor_parameter in zip(self.actor.parameters(), self.target_actor.parameters()):
            # overwrites the target network parameters with the following weighted sum
            target_actor_parameter.data.copy_(tau * actor_parameter.data + (1 - tau) * target_actor_parameter.data)

        # update target critic_1 weights
        for critic_parameter, target_critic_parameter in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            # overwrites the target network parameters with the following weighted sum
            target_critic_parameter.data.copy_(tau * critic_parameter.data + (1 - tau) * target_critic_parameter.data)

        # update target critic_2 weights
        for critic_parameter, target_critic_parameter in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            # overwrites the target network parameters with the following weighted sum
            target_critic_parameter.data.copy_(tau * critic_parameter.data + (1 - tau) * target_critic_parameter.data)

    def choose_action(self, observation: torch.Tensor, evaluate: bool = False):
        """
        Given the environment state, the agent chooses an action from its policy.

        Args:
            observation (torch.Tensor): The environment observation observed by the agent.

            evaluate (bool): A flag to turn noise on and off. During training, noise is added to the action for state space exploration. During evaluation, noise is not necessary for testing. (Seeing how the trained agent performs deterministically without noise during evaluation.)
        """
        # send the  observation torch tensor to the same device for computations
        observation = observation.to(self.device)

        # get the action components without noise added (forward pass through the actor network)
        mu = self.actor.forward(observation).to(self.actor.device)

        # add some noise to the deterministic action for the exploration of state space (explore-exploit dilemma)
        noise = torch.tensor(
            np.random.normal(scale=self.action_noise_std, size=self.action_size),
            dtype=torch.float,
            device=self.actor.device
        )
        noise *= torch.tensor(1 - int(evaluate))

        # bound the action (with added noise) between the min and max possible values
        mu = torch.clamp(
            mu + noise,
            self.action_min,
            self.action_max
        )

        return mu.cpu().detach().numpy()

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
        torch.save(obj=state_dictionaries['target_actor'],
                   f=os.path.join(directory, self.parameter_filenames['target_actor']))
        torch.save(obj=state_dictionaries['target_critic_1'],
                   f=os.path.join(directory, self.parameter_filenames['target_critic_1']))
        torch.save(obj=state_dictionaries['target_critic_2'],
                   f=os.path.join(directory, self.parameter_filenames['target_critic_2']))

    def load_parameters(self, directory: str):
        """
        Loads all the network parameters from files in the directory.
        """
        # get the state dictionaries
        state_dictionaries = {
            'actor': torch.load(os.path.normpath(os.path.join(directory, self.parameter_filenames['actor']))),
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

    @staticmethod
    def _concatenate_critic_state_batch(observations_batch: list[torch.tensor], actions_batch: list[torch.Tensor]):
        """
        This function will concatenate all agent observations and actions for a batch of experiences.

        observation indexing: [agent_index][batch sample, values]
        action indexing: [agent_index][batch sample, values]

        I will admit that this took me like 1.5 hours to figure out.
        The best thing I can do is include a code example here.


        Args:
            observations_batch:
            actions_batch:
        """
        critic_state_batch = torch.cat(
            [
                torch.cat(observations_batch, dim=1),
                torch.cat(actions_batch, dim=1)
            ],
            dim=1
        )
        return critic_state_batch

    def learn(
            self,
            agent_list,
            agent_index: int,
            observations: list[torch.Tensor],
            actions: list[torch.Tensor],
            rewards: list[torch.Tensor],
            next_observations: list[torch.Tensor],
            terminations: list[torch.Tensor]
    ):
        """
        Perform a learning/optimization step.

        Args:
            agent_list: A list of all the reinforcement learning agents in the algorithm.
            agent_index (int): The index in the lists of the agent which is currently learning.
            observations (list[torch.Tensor]): The batch of observations for the agents in the multi-agent algorithm.
            actions (list[torch.Tensor]): The batch of actions for the agents in the multi-agent algorithm.
            rewards (list[torch.Tensor]): The batch of rewards for the agents in the multi-agent algorithm.
            next_observations (list[torch.Tensor]): The batch of next_observations for the agents in the multi-agent algorithm.
            terminations (list[torch.Tensor]): The batch of terminations for the agents in the multi-agent algorithm.
        """
        # get the number of agents
        number_of_agents = len(agent_list)

        # send all the lists of tensors to the correct device (GPU/CPU)
        observations = [observations[index].to(self.device) for index in range(number_of_agents)]
        actions = [actions[index].to(self.device) for index in range(number_of_agents)]
        rewards = [rewards[index].to(self.device) for index in range(number_of_agents)]
        next_observations = [next_observations[index].to(self.device) for index in range(number_of_agents)]
        terminations = [terminations[index].to(self.device) for index in range(number_of_agents)]

        """ calculate the target actions with noise and clamping """
        # all agents' next actions are calculated for the batch
        target_actions = [
            # have to clamp the continuous action values again to what is allowed by the environment
            torch.clamp(
                # perform smoothing by adding some noise to the actions
                input=agent.target_actor.forward(next_observations[index]) + torch.clamp(torch.tensor(np.random.normal(scale=0.1)), -0.5, 0.5),
                min=self.action_min,
                max=self.action_max
            )
            for index, agent in enumerate(agent_list)
        ]

        """ get the critic values for the state-action pairs """
        # concatenate the target critic states
        critic_next_states = TD3._concatenate_critic_state_batch(
            observations_batch=next_observations,
            actions_batch=target_actions
        )

        # calculate the batch of q-values using the two target critics (for the next critic states)
        q1_ = self.target_critic_1.forward(critic_next_states).squeeze()
        q2_ = self.target_critic_2.forward(critic_next_states).squeeze()

        # concatenate the critic states
        critic_state = TD3._concatenate_critic_state_batch(
            observations_batch=observations,
            actions_batch=actions
        )

        # calculate the batch of q-values using the two online critics (for the current critic states)
        q1 = self.critic_1.forward(critic_state).squeeze()
        q2 = self.critic_2.forward(critic_state).squeeze()

        """ make use of the done mask to set terminal states to have values of 0.0 """
        q1_[terminations[agent_index]] = 0.0
        q2_[terminations[agent_index]] = 0.0

        # reshape the target critic values
        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        # take only the minimum values of the target critic values
        critic_value_ = torch.min(q1_, q2_)

        # calculate the TD target values (these are the predicted values/returns of the next states)
        target = rewards[agent_index] + self.discount_factor * critic_value_
        target = target.view(-1)  # reshape

        # zero gradients before performing backpropagation for the critic networks
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        # calculate the critic networks' losses
        q1_loss = torch.nn.functional.mse_loss(target, q1)
        q2_loss = torch.nn.functional.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss

        # perform the backpropagation step to calculate gradients
        critic_loss.backward()

        # perform optimization step using calculated gradients
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        """ TD3 delayed update of the actor and target networks """
        if self.optimization_step_counter % self.delay_interval == 0:
            # zero the actor network gradients
            self.actor.optimizer.zero_grad()


            # concatenate the critic states
            critic_state = TD3._concatenate_critic_state_batch(
                observations_batch=observations,
                actions_batch=mu
            )

            # calculate the actor loss
            actor_q1_loss = self.critic_1.forward(critic_state).squeeze()
            actor_loss = -torch.mean(actor_q1_loss)

            # perform backpropagation and calculate the gradients for the actor network
            actor_loss.backward()

            # use the gradients and perform an optimization step for the actor network
            self.actor.optimizer.step()

            # update the target network parameters
            self.update_parameters(tau=self.tau)

        # an optimization step just finished
        # increment the optimization step counter used for timing target network updates
        self.optimization_step_counter += 1
