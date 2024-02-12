"""
Description:
    The critic network for a LSTM-TD3 reinforcement learning agent.

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
import torch
from copy import deepcopy


class CriticNetwork(torch.nn.Module):
    """ The critic network for a LSTM-TD3 agent. """
    def __init__(
            self,
            observation_shape,
            action_shape,
            mem_pre_lstm_hid_sizes: tuple = (128,),
            mem_lstm_hid_sizes: tuple = (128,),
            mem_after_lstm_hid_size: tuple = (128,),
            cur_feature_hid_sizes: tuple = (128,),
            post_comb_hid_sizes: tuple = (128,),
            hist_with_past_act: bool = False
    ):
        """ Initializes the critic network for a LSTM-TD3 agent. """

        super(CriticNetwork, self).__init__()

        # save the passed in arguments
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.hist_with_past_act = hist_with_past_act

        # define the layers as module lists
        self.mem_pre_lstm_layers = torch.nn.ModuleList()
        self.mem_lstm_layers = torch.nn.ModuleList()
        self.mem_after_lstm_layers = torch.nn.ModuleList()
        self.cur_feature_layers = torch.nn.ModuleList()
        self.post_combined_layers = torch.nn.ModuleList()

        """ Layers: Pre-LSTM """

        # calculate layer size
        if self.hist_with_past_act:
            mem_pre_lstm_layer_size = [observation_shape + action_shape] + list(mem_pre_lstm_hid_sizes)
        else:
            mem_pre_lstm_layer_size = [observation_shape] + list(mem_pre_lstm_hid_sizes)

        # define the layers
        for h in range(len(mem_pre_lstm_layer_size) - 1):
            self.mem_pre_lstm_layers += [
                torch.nn.Linear(mem_pre_lstm_layer_size[h], mem_pre_lstm_layer_size[h + 1]),
                torch.nn.ReLU()
            ]

        """ Layers: LSTM """

        # calculate layer size
        self.mem_lstm_layer_sizes = [mem_pre_lstm_layer_size[-1]] + list(mem_lstm_hid_sizes)

        # define the layers
        for h in range(len(self.mem_lstm_layer_sizes) - 1):
            self.mem_lstm_layers += [
                torch.nn.LSTM(self.mem_lstm_layer_sizes[h], self.mem_lstm_layer_sizes[h + 1], batch_first=True)
            ]

        """ Layers: After-LSTM """

        # calculate layer size
        self.mem_after_lstm_layer_size = [self.mem_lstm_layer_sizes[-1]] + list(mem_after_lstm_hid_size)

        # define the layers
        for h in range(len(self.mem_after_lstm_layer_size)-1):
            self.mem_after_lstm_layers += [
                torch.nn.Linear(self.mem_after_lstm_layer_size[h], self.mem_after_lstm_layer_size[h+1]),
                torch.nn.ReLU()
            ]

        """ Layers: Current Feature Extraction """

        # calculate layer size
        cur_feature_layer_size = [observation_shape + action_shape] + list(cur_feature_hid_sizes)

        # define the layers
        for h in range(len(cur_feature_layer_size) - 1):
            self.cur_feature_layers += [
                torch.nn.Linear(cur_feature_layer_size[h], cur_feature_layer_size[h + 1]),
                torch.nn.ReLU()
            ]

        """ Layers: Post-Combination """

        # calculate layer size
        post_combined_layer_size = [self.mem_after_lstm_layer_size[-1] + cur_feature_layer_size[-1]] + list(
            post_comb_hid_sizes) + [1]

        # define the layers
        for h in range(len(post_combined_layer_size) - 2):
            self.post_combined_layers += [
                torch.nn.Linear(post_combined_layer_size[h], post_combined_layer_size[h + 1]),
                torch.nn.ReLU()
            ]
        self.post_combined_layers += [
            torch.nn.Linear(post_combined_layer_size[-2], post_combined_layer_size[-1]),
            torch.nn.Identity()
        ]

    def forward(self, observation, action, history_observations, history_actions, history_segment_length: torch.Tensor):
        """
        Perform the forward pass through the actor network, which includes a LSTM.

        Args:
            observation: The current observation of the agent in the environment.

            action: The action the agent took given the current observation in the environment.

            history_observations: The history of observations up until the current observation.

            history_actions: The history of actions taken up until the current action taken.

            history_segment_length (torch.Tensor): The number of observation-action pairs in the history

        Returns:
            Q-value of the current state given the history
        """

        """ Determine if the history needs to have the observation and actions concatenated. """
        # checking the length of the history being used
        tmp_hist_seg_len = deepcopy(history_segment_length)
        tmp_hist_seg_len[history_segment_length == 0] = 1

        # concatenate the observation and action (or just use the observation)
        if self.hist_with_past_act:
            x = torch.cat([history_observations, history_actions], dim=-1)
        else:
            x = history_observations

        """ Layers: Pre-LSTM """
        for layer in self.mem_pre_lstm_layers:
            x = layer(x)

        """ Layers: LSTM """
        for layer in self.mem_lstm_layers:
            x, (lstm_hidden_state, lstm_cell_state) = layer(x)

        """ Layers: After-LSTM """
        for layer in self.mem_after_lstm_layers:
            x = layer(x)

        # History output mask to reduce disturbance caused by none history memory
        #
        # Jordan Note:
        # Alright, this part is super confusing.
        # https://pytorch.org/docs/stable/generated/torch.gather.html
        extracted_memory = torch.gather(
            input=x,
            dim=1,
            index=(tmp_hist_seg_len - 1).view(-1, 1).repeat(1, self.mem_after_lstm_layer_size[-1]).unsqueeze(1).long()
        ).squeeze(1)

        """ Layers: Current Feature Extraction """
        x = torch.cat([observation, action], dim=-1)
        for layer in self.cur_feature_layers:
            x = layer(x)

        """ Layers: Post-Combination """
        x = torch.cat([extracted_memory, x], dim=-1)

        for layer in self.post_combined_layers:
            x = layer(x)

        # squeeze(x, -1) : critical to ensure q has right shape.
        x = torch.squeeze(x, -1)

        return x, extracted_memory
