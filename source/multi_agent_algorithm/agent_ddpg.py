import os
from copy import deepcopy
import torch
import numpy as np
from source.multi_agent_algorithm.critic_network import CriticNetwork
from source.multi_agent_algorithm.actor_network import ActorNetwork

class DDPG:
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
            tau: float = 0.01
    ):
 """
        Args:
            name (str): A name given to the agent.

            observation_size (int): The observation size to the actor network. The size is just the size of the individual agent's observation.

            critic_state_size (int): The state size to the critic network. The state size is the size of all agents states and actions combined (For the MADDPG algorithm).

            action_size (int): The number of action components to output from the actor network.

            action_min (float): The minimum value/bound for an action component.

            action_max (float): The maximum value/bound for an action component.

            action_noise_std (float): For DDPG agents, noise is added to their actions. This action_noise parameter value is the standard deviation for a normal distribution, mean=0, that the noise is sampled from.

            actor_learning_rate (float): The learning rate of the actor network.

            critic_learning_rate (float): The learning rate of the critic network.

            actor_layer_1_size (int): The output size of the first fully connected linear layer.

            actor_layer_2_size (int): The output size of the second fully connected linear layer.

            critic_layer_1_size (int): The output size of the first fully connected linear layer.

            critic_layer_2_size (int): The output size of the second fully connected linear layer.

            discount_factor (float): The discount factor for discounted returns.

            tau (float): tau is the amount of the online weights that will be copied over to the target network, while (1-tau) is the amount of the target weights that will remain the same.
        """
        # save passed in arguments
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

        # filenames for saving and loading model parameters (DDPG)
        self.parameter_filenames = {
            'actor': 'actor.pt',
            'critic': 'critic.pt',
            'target_actor': 'target_actor.pt',
            'target_critic': 'target_critic.pt',
        }
        # create the online actor and critic networks
        self.actor = ActorNetwork(
            learning_rate=actor_learning_rate,
            action_size=action_size,
            action_min=action_min,
            action_max=action_max,
            input_size=observation_size,
            layer_1_size=actor_layer_1_size,
            layer_2_size=actor_layer_2_size
        )
        self.critic = CriticNetwork(
            learning_rate=critic_learning_rate,
            input_size=critic_state_size,
            layer_1_size=critic_layer_1_size,
            layer_2_size=critic_layer_2_size
        )

        # create the target actor and critic networks
        # note:
        # deepcopy() will also ensure that the weights are the same for the target networks
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

        # send the networks to GPUs if available, else CPUs
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.actor.send_to_device(self.device)
        self.critic.send_to_device(self.device)
        self.target_actor.send_to_device(self.device)
        self.target_critic.send_to_device(self.device)

        # report that the agent was successfully created
        print(f'... DDPG Agent "{self.name}" Created Successfully ...')
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

        # update target critic weights
        for critic_parameter, target_critic_parameter in zip(self.critic.parameters(), self.target_critic.parameters()):
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

        # get the actions from an actor network forward pass
        actions = self.actor.forward(observation)

        # prepare noise to add to the action during training
        # no noise is added during model evaluation
        noise = torch.tensor(
            np.random.normal(scale=self.action_noise_std, size=self.action_size),
            dtype=torch.float,
            device=self.actor.device
        )
        noise *= torch.tensor(1 - int(evaluate))

        # bound the action (with added noise) between the min and max possible values
        action = torch.clamp(
            actions + noise,
            self.action_min,
            self.action_max
        )

        # send the torch tensor to the CPU and convert it to a numpy array
        # print('action', action)
        # print('conversion', action.data.cpu().numpy())
        return action.cpu().detach().numpy()
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
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
        }

        # save (actor, critic_1, critic_2) (TD3)
        torch.save(obj=state_dictionaries['actor'], f=os.path.join(directory, self.parameter_filenames['actor']))
        torch.save(obj=state_dictionaries['critic'], f=os.path.join(directory, self.parameter_filenames['critic']))

        # save (target_actor, target_critic_1, target_critic_2) (TD3)
        torch.save(obj=state_dictionaries['target_actor'],
                   f=os.path.join(directory, self.parameter_filenames['target_actor']))
        torch.save(obj=state_dictionaries['target_critic'],
                   f=os.path.join(directory, self.parameter_filenames['target_critic']))

    def load_parameters(self, directory: str):
        """
        Loads all the network parameters from files in the directory.
        """
        # get the state dictionaries
        state_dictionaries = {
            'actor': torch.load(os.path.normpath(os.path.join(directory, self.parameter_filenames['actor']))),
            'critic': torch.load(os.path.join(directory, self.parameter_filenames['critic'])),
            'target_actor': torch.load(os.path.join(directory, self.parameter_filenames['target_actor'])),
            'target_critic': torch.load(os.path.join(directory, self.parameter_filenames['target_critic']))
        }

        # load (actor, critic_1, critic_2) (TD3)
        self.actor.load_state_dict(state_dict=state_dictionaries['actor'])
        self.critic.load_state_dict(state_dict=state_dictionaries['critic'])

        # load (target_actor, target_critic_1, target_critic_2) (TD3)
        self.target_actor.load_state_dict(state_dict=state_dictionaries['target_actor'])
        self.target_critic.load_state_dict(state_dict=state_dictionaries['target_critic'])

    @staticmethod
    def _concatenate_critic_state_batch(observations_batch: list[torch.tensor], actions_batch: list[torch.Tensor]):
        """
        This function will concatenate all agent observations and actions for a batch of experiences.

        observation indexing: [agent_index][batch sample, values]
        action indexing: [agent_index][batch sample, values]


        Example:
            >>> # creating a sample batch to concatenate
            >>> observation_sizes = [2, 3, 3]  # sizes could be different for each agent
            >>> action_sizes = [2, 2, 2]  # sizes could be different for each agent
            >>> number_of_agents = 3
            >>> batch_size = 5

            >>> low = 0  # random int lower bound
            >>> high = 10  # random int upper bound

            >>> observations_batch = [torch.randint(low=low, high=high, size=(batch_size, size)) for size in observation_sizes]
            >>> actions_batch = [torch.randint(low=low, high=high, size=(batch_size, size)) for size in action_sizes]

            >>> print('observations batch:', observations_batch)
            >>> print('actions batch:', actions_batch)

            >>> # super tricky
            >>> concatenated_observations_batch_tensor = torch.cat(observations_batch, dim=1)
            >>> concatenated_actions_batch_tensor = torch.cat(actions_batch, dim=1)
            >>> concatenated_critic_state_tensor = torch.cat([concatenated_observations_batch_tensor, concatenated_actions_batch_tensor], dim=1)

            >>> # observations_batch[agent_number][sample_number]
            >>> # actions_batch[agent_number][sample_number]
            >>> print('concatenated observations batch =',  concatenated_observations_batch_tensor)
            >>> print('concatenated actions batch =',  concatenated_actions_batch_tensor)
            >>> print('state1 =', [observations_batch[x][0] for x in range(number_of_agents)], [actions_batch[x][0] for x in range(number_of_agents)])
            >>> print(concatenated_critic_state_tensor)

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

        # from the next_observations, calculate all agents next actions and next critic values
        # it is not necessary to calculate gradients, use torch.no_grad() to reduce computational cost
        with torch.no_grad():
            # all agents' next actions are calculated for the batch and concatenated
            next_actions = [agent.target_actor.forward(next_observations[index]) for index, agent in enumerate(agent_list)]

            # batch of concatenated critic next states
            critic_next_states = DDPG._concatenate_critic_state_batch(
                observations_batch=next_observations,
                actions_batch=next_actions
            )

            # all agents' next critic values are calculated
            critic_value_ = self.target_critic.forward(critic_next_states).squeeze()

            # the value of a terminal state is 0.0
            critic_value_[terminations[agent_index]] = 0.0

            # the one step temporal difference (bootstrapped) target return value
            target = rewards[agent_index] + self.discount_factor * critic_value_

        # batch of concatenated critic states
        critic_states = DDPG._concatenate_critic_state_batch(
            observations_batch=observations,
            actions_batch=actions
        )

        # critic value for the current state is calculated
        critic_value = self.critic.forward(critic_states).squeeze()

        # calculate the critic loss
        critic_loss = torch.nn.functional.mse_loss(target, critic_value)

        # perform backpropagation for the critic to calculate gradients for the network weights
        self.critic.optimizer.zero_grad()
        critic_loss.backward()

        # clip the gradients so that the optimization step is bounded
        # prevents making too large a change to the network parameters at one time
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)

        # update the online critic parameters
        self.critic.optimizer.step()

        # calculating action for the current agent again (a computational graph is needed for backpropagation)
        actions[agent_index] = self.actor.forward(observations[agent_index])

        # batch of concatenated critic states
        critic_states = DDPG._concatenate_critic_state_batch(
            observations_batch=observations,
            actions_batch=actions
        )

        # calculate the actor loss
        actor_loss = -self.critic.forward(critic_states).mean()

        # perform backpropagation for the actor to calculate gradients for the network weights
        self.actor.optimizer.zero_grad()
        actor_loss.backward()

        # clip the gradients so that the optimization step is bounded
        # prevents making too large a change to the network parameters at one time
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)

        # update the online actor parameters
        self.actor.optimizer.step()

        # update the target network parameters
        self.update_parameters(tau=self.tau)
