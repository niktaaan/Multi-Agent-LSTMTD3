"""
Description:
    A generic actor network for an Actor-Critic reinforcement learning agent is coded here.

Author:
    Jordan Cramer

Date:
    2023-08-22
"""
import os
import torch


class ActorNetwork(torch.nn.Module):
    """
    The actor network for a reinforcement learning agent.
    The actor only gets its observation information.
    """
    def __init__(
            self,
            learning_rate: float,
            action_size: int,
            action_min: float,
            action_max: float,
            input_size: int,
            layer_1_size: int,
            layer_2_size: int
    ):
        """
        Args:
            learning_rate (float): The learning rate of the actor network.

            action_size (int): The number of action components to output from the network.

            action_min (float): The minimum value/bound for an action component.

            action_max (float): The maximum value/bound for an action component.

            input_size (int): The input size to the network. The size is just the size of the individual agent's observation/state.

            layer_1_size (int): The output size of the first fully connected linear layer.

            layer_2_size (int): The output size of the second fully connected linear layer.
        """
        super(ActorNetwork, self).__init__()

        # save the passed in arguments
        self.learning_rate = learning_rate
        self.action_size = action_size
        self.action_min = action_min
        self.action_max = action_max
        self.input_size = input_size
        self.layer_1_size = layer_1_size
        self.layer_2_size = layer_2_size

        # define the layers
        self.layer_1 = torch.nn.Linear(input_size, layer_1_size)
        self.layer_2 = torch.nn.Linear(layer_1_size, layer_2_size)
        self.action = torch.nn.Linear(layer_2_size, action_size)

        # set up the optimizer
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # device
        self.device = 'cpu'

    def send_to_device(self, device):
        """ Sends the model to a GPU if available for computations, else CPU. """
        self.device = device
        self.to(self.device)

    def forward(self, observation: torch.Tensor):
        """
        The forward pass to calculate an action for the agent.

        Args:
            observation (torch.Tensor): The agent's observation at one time step. Expected to already be on the correct device for computations.

            Returns:
                The agent's action.
        """
        # forward propagation through the base layers
        x = torch.nn.functional.relu(self.layer_1(observation))
        x = torch.nn.functional.relu(self.layer_2(x))
        x = self.action(x)

        # activate the action output and bound it within the action space

        # bounded between (-1,1)
        x = torch.nn.functional.tanh(x)

        # bounded between (self.action_min, self.action_max)
        # note:
        # I'm not sure if this is the best way to bound the output
        # I'm not sure if it affects further steps for DDPG or TD3 algorithms (operations like clamping)
        action = ((x + 1.0) / 2.0) * (self.action_max - self.action_min) + self.action_min

        return action
