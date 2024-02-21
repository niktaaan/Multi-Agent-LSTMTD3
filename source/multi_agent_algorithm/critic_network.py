import os
import torch


class CriticNetwork(torch.nn.Module):

    def __init__(
            self,
            learning_rate: float,
            input_size: int,
            layer_1_size: int,
            layer_2_size: int
    ):
        """
        Args:
            learning_rate (float): The learning rate of the critic network.

            input_size (int): The input size to the network. The input size is the size of all agents states and actions combined (For the MADDPG algorithm).

            layer_1_size (int): The output size of the first fully connected linear layer.

            layer_2_size (int): The output size of the second fully connected linear layer.
        """
        super(CriticNetwork, self).__init__()

        # save the passed in arguments
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.layer_1_size = layer_1_size
        self.layer_2_size = layer_2_size

        # define the layers
        self.layer_1 = torch.nn.Linear(input_size, layer_1_size)
        self.layer_2 = torch.nn.Linear(layer_1_size, layer_2_size)
        self.q_value = torch.nn.Linear(layer_2_size, 1)

        # set up the optimizer
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # device
        self.device = 'cpu'

    def send_to_device(self, device):
        """ Sends the model to a GPU if available for computations, else CPU. """
        self.device = device
        self.to(self.device)

    def forward(
            self,
            state: torch.Tensor
    ):
        """
        The forward pass to calculate a Q-value for the states and actions of all agents in the network.

        Args:
            state (torch.Tensor): The concatenated observations and actions of all agents in the environment at one time step. Expected to already be on the correct device for computations.

        Returns:
                The Q-value.
        """
        # forward propagation through the base layers
        x = torch.nn.functional.relu(self.layer_1(state))
        x = torch.nn.functional.relu(self.layer_2(x))

        # final Q-value output
        q_value = self.q_value(x)

        return q_value
