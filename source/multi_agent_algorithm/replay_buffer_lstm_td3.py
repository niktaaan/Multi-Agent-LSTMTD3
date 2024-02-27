import numpy as np
import torch


class HistoryBufferLSTMTD3:

    def __init__(
            self,
            observation_size: int,
            action_size: int,
            max_history_length: int
    ):
        # save passed in arguments
        self.observation_size = observation_size
        self.action_size = action_size
        self.max_history_length = max_history_length

        # creating buffers to store (observation,action) pairs
        # (the max_history_length will always be greater than 0 (i.e. histories will always be used for LSTM-TD3))
        self.observation_buffer = torch.zeros(size=[self.max_history_length, self.observation_size], dtype=torch.float)
        self.action_buffer = torch.zeros(size=[self.max_history_length, self.action_size], dtype=torch.float)
        self.observation_buffer_length = 0

    def episode_reset(self, starting_observation: np.ndarray):
        """
        The history buffer needs to be reset for new episodes.
        During a new episode, a new (observation,action) pair history will start.
        """
        self.observation_buffer = torch.zeros(size=[self.max_history_length, self.observation_size], dtype=torch.float)
        self.action_buffer = torch.zeros(size=[self.max_history_length, self.action_size], dtype=torch.float)
        self.observation_buffer[0, :] = torch.tensor(data=starting_observation, dtype=torch.float)
        self.observation_buffer_length = 0

    def add_observation_action(self, observation: np.ndarray, action: np.ndarray):
        """
        Add a new (observation,action) pair to the history.
        Environments like PettingZoo will give observations/actions as numpy.ndarrays.
        """
        # when the number of (observation,action) pairs has reached the max history length
        if self.observation_buffer_length == self.max_history_length:
            # shift the history pairs over to make room for one more pair at the end
            self.observation_buffer[:self.max_history_length - 1] = self.observation_buffer[1:].clone()
            self.action_buffer[:self.max_history_length - 1] = self.action_buffer[1:].clone()

            # add the new (observation,action) pair to the end of the buffers
            self.observation_buffer[self.max_history_length - 1] = torch.tensor(observation, dtype=torch.float)
            self.action_buffer[self.max_history_length - 1] = torch.tensor(action, dtype=torch.float)

        # the history buffer still has room for some more (observation,action) pairs
        else:
            self.observation_buffer[self.observation_buffer_length] = torch.tensor(observation, dtype=torch.float)
            self.action_buffer[self.observation_buffer_length] = torch.tensor(action, dtype=torch.float)
            self.observation_buffer_length += 1


class ReplayBufferLSTMTD3:
    """ An experience replay buffer for a single LSTM-TD3 agent. """
    def __init__(
            self,
            observation_size: int,
            action_size: int,
            buffer_size: int,
            batch_size: int,
            max_history_length: int
    ):
        # save passed in arguments
        self.observation_size = observation_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.max_history_length = max_history_length

        # create a history buffer to keep track of (observation,action) pair histories during episodes
        self.history_buffer = HistoryBufferLSTMTD3(
            observation_size=self.observation_size,
            action_size=self.action_size,
            max_history_length=self.max_history_length
        )

        # create storage for the replay buffer
        self.observation_memory = torch.zeros(size=[self.buffer_size, self.observation_size], dtype=torch.float)
        self.next_observation_memory = torch.zeros(size=[self.buffer_size, self.observation_size], dtype=torch.float)
        self.action_memory = torch.zeros(size=[self.buffer_size, self.action_size], dtype=torch.float)
        self.reward_memory = torch.zeros(size=[self.buffer_size], dtype=torch.float)
        self.terminal_memory = torch.zeros(size=[self.buffer_size], dtype=torch.float)

        # keep track of
        # (1) the current index to store new experiences at
        # (2) the current number of stored experiences
        self.memory_index = 0
        self.memory_count = 0

    def episode_reset(self, starting_observation: np.ndarray):
        """
        The history buffer needs to be reset for new episodes.
        During a new episode, a new (observation,action) pair history will start.
        """
        self.history_buffer.episode_reset(starting_observation=starting_observation)

    def store_experience(
            self,
            observation: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            next_observation: np.ndarray,
            termination: np.ndarray
    ):
        """ Store an experience in the replay buffer. """
        # store all the experience information in the buffer at the current memory index
        self.observation_memory[self.memory_index] = torch.tensor(observation, dtype=torch.float)
        self.action_memory[self.memory_index] = torch.tensor(action, dtype=torch.float)
        self.reward_memory[self.memory_index] = torch.tensor(reward, dtype=torch.float)
        self.next_observation_memory[self.memory_index] = torch.tensor(next_observation, dtype=torch.float)
        self.terminal_memory[self.memory_index] = torch.tensor(termination, dtype=torch.float)

        # calculate the next replay buffer index to store an experience
        self.memory_index = (self.memory_index + 1) % self.buffer_size

        # the number of currently stored experiences is either,
        # (1) the number of stored memories (if it is smaller than the max buffer size)
        # (2) the max buffer size (when the replay buffer is completely filled)
        self.memory_count = min(self.memory_count + 1, self.buffer_size)

        # add the most recent (observation,action) pair to the history buffer
        self.history_buffer.add_observation_action(
            observation=observation,
            action=action
        )

    def sample_buffer(self, indices: np.ndarray):
        """
        Sample a batch of experiences uniformly from the replay buffer.
        These include observation-action pair histories.

        Args:
            indices (np.ndarray): Random indices for experiences to sample.

        Returns:
            A dictionary containing the batch of experiences, also including (observation,action) pair histories.
            All values are stored as torch tensors.
        """
        # create storage for batches of histories
        #
        # h_o: history of observations
        # h_o2: history of next observations
        #
        # h_a: history of actions
        # h_a2: history of next actions
        #
        # h_o_l: history observations length (the number of (observation,action) pairs in the histories)
        # h_o2_l: history next observations length (the number of (observation,action) pairs in the histories)
        h_o = torch.zeros(size=[self.batch_size, self.max_history_length, self.observation_size], dtype=torch.float)
        h_o2 = torch.zeros(size=[self.batch_size, self.max_history_length, self.observation_size], dtype=torch.float)

        h_a = torch.zeros(size=[self.batch_size, self.max_history_length, self.action_size], dtype=torch.float)
        h_a2 = torch.zeros(size=[self.batch_size, self.max_history_length, self.action_size], dtype=torch.float)

        h_o_l = self.max_history_length * torch.ones(size=[self.batch_size])
        h_o2_l = self.max_history_length * torch.ones(size=[self.batch_size])

        # for: each sample in the batch
        # get the histories of (observation,action) pairs
        #
        # histories:
        # a history includes all the (observation,action) pairs before the index
        # up to a total number of mox_history_length pairs
        for i, index in enumerate(indices):

            # simple case:
            # determine the starting index in the buffer for the history

            start_index = index - self.max_history_length
            if start_index < 0:
                start_index = 0

            # special case:
            # around the start of a new episode when the current history is not yet max_history_length,
            # if the history terminates before the last experience (not including a terminal flag in index),
            # start from the index next to the termination
            

            terminal_indices: list[torch.Tensor] = torch.where(self.terminal_memory[start_index:index] == 1)
            terminal_found: bool = len(terminal_indices[0]) != 0
            if terminal_found:
                # start one index after the terminal flag index
                start_index = start_index + terminal_indices[0][-1] + 1

            # figure out the history length (number of (observation,action) pairs in the history)
            h_l = index - start_index
            h_o_l[i] = h_l

            # get the histories of (observation,action) pairs
            h_o[i, :h_l, :] = self.observation_memory[start_index:index]
            h_a[i, :h_l, :] = self.action_memory[start_index:index]

            # special case:
            # if the first experience of an episode is sampled (there is no (observation,action) history yet)
            # the history lengths are different for observations/actions (0) and next_observations/next_actions (1)
            if h_l == 0:
                h_o2_l[i] = 1
            else:
                h_o2_l[i] = h_l

            # get the histories for the (next_observation,next_action) pairs
            h_o2[i, :h_l, :] = self.next_observation_memory[start_index:index]
            h_a2[i, :h_l, :] = self.action_memory[start_index + 1:index + 1]

        # the batch of samples
        #
        # o: observations
        # o2: next observations
        # a: actions
        # r: rewards
        # t: terminal
        # h_o: history of observations
        # h_o2: history of next observations
        # h_a: history of actions
        # h_a2: history of next actions
        # h_o_l: history observations length (the number of (observation,action) pairs in the histories)
        # h_o2_l: history next observations length (the number of (observation,action) pairs in the histories)
        batch = dict(
            o=self.observation_memory[indices],
            o2=self.next_observation_memory[indices],
            a=self.action_memory[indices],
            r=self.reward_memory[indices],
            t=self.terminal_memory[indices],
            h_o=h_o,
            h_o2=h_o2,
            h_a=h_a,
            h_a2=h_a2,
            h_o_l=h_o_l,
            h_o2_l=h_o2_l
        )

        # return the batch of experiences with observation-action pair histories
        return batch
