from copy import deepcopy
import torch


class ActorLSTMTD3(torch.nn.Module):
    """ The actor network for a LSTM-TD3 agent. """
    def __init__(
            self,
            observation_size: int,
            action_size: int,
            action_min: float,
            action_max: float,
            mem_pre_lstm_hid_sizes: tuple | list[int] = (128,),
            mem_lstm_hid_sizes: tuple | list[int] = (128,),
            mem_after_lstm_hid_size: tuple | list[int] = (128,),
            cur_feature_hid_sizes: tuple | list[int] = (128,),
            post_comb_hid_sizes: tuple | list[int] = (128,)
    ):
        """
        Initializes the actor network for a LSTM-TD3 agent.

        Args:
            observation_size (int): The number of floating point numbers in the observation vector.
            action_size (int): The number of floating point numbers (action components) to output from the network.
            action_min (float): The lower bound value for an output action component.
            action_max (float): The upper bound value for an output action component.
        """
        # set up torch.nn.Module
        super(ActorLSTMTD3, self).__init__()

        # save passed in arguments
        self.observation_size = observation_size
        self.action_size = action_size
        self.action_min = action_min
        self.action_max = action_max

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
        #
        # explanation:
        # lists are being created here
        # example: my_list = [10,20,15]
        #
        # the list is then used to create an architecture like the following
        # input_layer (input_size=10, output_size=20)
        # hidden_layer_1 (input_size=20, output_size=15)
        #
        # refer to the diagram of the architecture from the paper if confused
        # paper: https://arxiv.org/pdf/2102.12344.pdf
        self.mem_pre_lstm_layer_size = [self.observation_size + self.action_size] + mem_pre_lstm_hid_sizes
        self.mem_lstm_layer_sizes = [self.mem_pre_lstm_layer_size[-1]] + mem_lstm_hid_sizes
        self.mem_after_lstm_layer_size = [self.mem_lstm_layer_sizes[-1]] + mem_after_lstm_hid_size
        self.cur_feature_layer_size = [self.observation_size] + cur_feature_hid_sizes
        self.post_combined_layer_size = [self.mem_after_lstm_layer_size[-1] + self.cur_feature_layer_size[-1]] + post_comb_hid_sizes + [action_size]

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
                torch.nn.Tanh()
            ]

        # build the architecture
        build_pre_lstm_layers()
        build_lstm()
        build_after_lstm_layers()
        build_current_feature_extraction_layers()
        build_post_combination_layers()

    def forward(
            self,
            o: torch.cuda.FloatTensor,
            h_o: torch.cuda.FloatTensor,
            h_a: torch.cuda.FloatTensor,
            h_l: torch.cuda.FloatTensor
    ):
        """
        Perform the forward pass through the actor network, which includes a LSTM.

        Args:
            o (torch.cuda.FloatTensor): The current observation of the agent in the environment. (expected to be on the GPU device)
            h_o (torch.cuda.FloatTensor): The history of observations up until the current observation. (expected to be on the GPU device)
            h_a (torch.cuda.FloatTensor): The history of actions taken up until the current action taken. (expected to be on the GPU device)
            h_l (torch.cuda.FloatTensor): The number of (observation,action) pairs in the history. (expected to be on the GPU device)

        Returns:
            Action for the agent.
        """

        """ Concatenate the (observation,action) history pairs """
        tmp_h_l = deepcopy(h_l)
        tmp_h_l[h_l == 0] = 1

        x = torch.cat([h_o, h_a], dim=-1)

        """ Layers: Pre-LSTM """
        for layer in self.mem_pre_lstm_layers:
            x = layer(x)

        """ Layers: LSTM """
        for layer in self.mem_lstm_layers:
            x, (lstm_hidden_state, lstm_cell_state) = layer(x)

        """ Layers: After-LSTM """
        for layer in self.mem_after_lstm_layers:
            x = layer(x)

        # History output mask to reduce disturbance caused by non history memory
        #
        # Jordan Note:
        # Alright, this part is super confusing.
        # https://pytorch.org/docs/stable/generated/torch.gather.html
        extracted_memory = torch.gather(
            input=x,
            dim=1,
            index=(tmp_h_l - 1).view(-1, 1).repeat(1, self.mem_after_lstm_layer_size[-1]).unsqueeze(1).long()
        ).squeeze(1)

        """ Layers: Current Feature Extraction """
        x = o
        for layer in self.cur_feature_layers:
            x = layer(x)

        """ Layers: Post-Combination """
        x = x.view(extracted_memory.shape[0], -1)
        x = torch.cat([extracted_memory, x], dim=-1)

        for layer in self.post_combined_layers:
            x = layer(x)

        # the final activation of the post_combined_layers is a tanh()
        # the value will be bounded between (-1,1)
        # the following line will scale these bounds to (-self.action_max, +self.action_max)
        # eventually, this value should be bounded by (self.action_min, self.action_max) according to the environment
        x = self.action_max * x

        return x, extracted_memory
