from copy import deepcopy
import torch


class CriticLSTMTD3(torch.nn.Module):

    def __init__(
            self,
            critic_state_size: int,
            mem_pre_lstm_hid_sizes: tuple | list[int] = (128,),
            mem_lstm_hid_sizes: tuple | list[int] = (128,),
            mem_after_lstm_hid_size: tuple | list[int] = (128,),
            cur_feature_hid_sizes: tuple | list[int] = (128,),
            post_comb_hid_sizes: tuple | list[int] = (128,)
    ):
        """

        Args:
            critic_state_size (int): The size of all the agents' observation and action sizes combined.
        """
        # set up torch.nn.Module
        super(CriticLSTMTD3, self).__init__()

        # save passed in arguments
        self.critic_state_size: int = critic_state_size

        # save passed in arguments (layer sizes)
        self.mem_pre_lstm_hid_sizes: list[int] = mem_pre_lstm_hid_sizes
        self.mem_lstm_hid_sizes: list[int] = mem_lstm_hid_sizes
        self.mem_after_lstm_hid_size: list[int] = mem_after_lstm_hid_size
        self.cur_feature_hid_sizes: list[int] = cur_feature_hid_sizes
        self.post_comb_hid_sizes: list[int] = post_comb_hid_sizes

        # define the layers as module lists
        self.mem_pre_lstm_layers = torch.nn.ModuleList()
        self.mem_lstm_layers = torch.nn.ModuleList()
        self.mem_after_lstm_layers = torch.nn.ModuleList()
        self.cur_feature_layers = torch.nn.ModuleList()
        self.post_combined_layers = torch.nn.ModuleList()

        # calculate layer sizes
       
        self.mem_pre_lstm_layer_size = [self.critic_state_size] + mem_pre_lstm_hid_sizes
        self.mem_lstm_layer_sizes = [self.mem_pre_lstm_layer_size[-1]] + mem_lstm_hid_sizes
        self.mem_after_lstm_layer_size = [self.mem_lstm_layer_sizes[-1]] + mem_after_lstm_hid_size
        self.cur_feature_layer_size = [self.critic_state_size] + cur_feature_hid_sizes
        self.post_combined_layer_size = [self.mem_after_lstm_layer_size[-1] + self.cur_feature_layer_size[-1]] + post_comb_hid_sizes + [1]

        # create the definitions for building the architecture
        # the size and number of layers can vary based on hyperparameters
        def build_pre_lstm_layers():
            # define the layers
            for h in range(len(self.mem_pre_lstm_layer_size) - 1):
                self.mem_pre_lstm_layers += [
                    torch.nn.Linear(self.mem_pre_lstm_layer_size[h], self.mem_pre_lstm_layer_size[h + 1]),
                    torch.nn.ReLU()
                ]

        def build_lstm():
            # define the layers
            for h in range(len(self.mem_lstm_layer_sizes) - 1):
                self.mem_lstm_layers += [
                    torch.nn.LSTM(self.mem_lstm_layer_sizes[h], self.mem_lstm_layer_sizes[h + 1], batch_first=True)
                ]

        def build_after_lstm_layers():
            # define the layers
            for h in range(len(self.mem_after_lstm_layer_size) - 1):
                self.mem_after_lstm_layers += [
                    torch.nn.Linear(self.mem_after_lstm_layer_size[h], self.mem_after_lstm_layer_size[h + 1]),
                    torch.nn.ReLU()
                ]

        def build_current_feature_extraction_layers():
            # define the layers
            for h in range(len(self.cur_feature_layer_size) - 1):
                self.cur_feature_layers += [
                    torch.nn.Linear(self.cur_feature_layer_size[h], self.cur_feature_layer_size[h + 1]),
                    torch.nn.ReLU()
                ]

        def build_post_combination_layers():
            # define the layers
            for h in range(len(self.post_combined_layer_size) - 2):
                self.post_combined_layers += [
                    torch.nn.Linear(self.post_combined_layer_size[h], self.post_combined_layer_size[h + 1]),
                    torch.nn.ReLU()
                ]
            self.post_combined_layers += [
                torch.nn.Linear(self.post_combined_layer_size[-2], self.post_combined_layer_size[-1]),
                torch.nn.Identity()
            ]

        # build the architecture
        build_pre_lstm_layers()
        build_lstm()
        build_after_lstm_layers()
        build_current_feature_extraction_layers()
        build_post_combination_layers()

    def forward(
            self,
            c_s:  torch.cuda.FloatTensor,
            h_c_s:  torch.cuda.FloatTensor,
            h_l:  torch.cuda.FloatTensor
    ):
        """
        Perform the forward pass through the critic network, which includes a LSTM.

        Args:
            c_s (torch.cuda.FloatTensor): The critic state. The concatenation of all agents' observations and actions. (expected to be on the GPU device)
            h_c_s (torch.cuda.FloatTensor): The history of critic states up until the current state. (expected to be on the GPU device)
            h_l (torch.cuda.FloatTensor): The number of critic states in the history. (expected to be on the GPU device)

        Returns:
            Q-value of the current state given the history
        """

        """ Determine if the history needs to have the observation and actions concatenated. """
        # checking the length of the history being used
        tmp_h_l = deepcopy(h_l)
        tmp_h_l[h_l == 0] = 1


        x = h_c_s

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
       
        extracted_memory = torch.gather(
            input=x,
            dim=1,
            index=(tmp_h_l - 1).view(-1, 1).repeat(1, self.mem_after_lstm_layer_size[-1]).unsqueeze(1).long()
        ).squeeze(1)

        """ Layers: Current Feature Extraction """
        """x = torch.cat([observation, action], dim=-1)"""
        x = c_s
        for layer in self.cur_feature_layers:
            x = layer(x)

        """ Layers: Post-Combination """
        x = torch.cat([extracted_memory, x], dim=-1)

        for layer in self.post_combined_layers:
            x = layer(x)

        # squeeze(x, -1) : critical to ensure q has right shape.
        x = torch.squeeze(x, -1)

        return x, extracted_memory
