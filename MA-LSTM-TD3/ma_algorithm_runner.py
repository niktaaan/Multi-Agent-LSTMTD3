"""
Description:
    The algorithm runner class is for simplifying the training of all algorithms.
    The class will be responsible for,

    1. Training the algorithms
    2. Rendering the algorithms
    3. Exporting graph and .csv file data

Author:
    Jordan Cramer

Date:
    2023-08-30

References:
    * MIT License
    * https://github.com/philtabor/Multi-Agent-Reinforcement-Learning/tree/main
    
    * MIT License
    * https://github.com/LinghengMeng/LSTM-TD3
"""
import os
from copy import deepcopy
import torch
import numpy as np
import json
import pygame
import time
from source.multi_agent_algorithm.create_algorithm import create_algorithm
from source.environment.create_environment import create_environment
from source.utility.dynamic_array import DynamicArray
from source.utility.data_manager import DataManager


class AlgorithmRunner:
    """ A class that can train multi-agent algorithms (MADDPG, MATD3, MALSTMTD3) and render them. """

    def __init__(self, arguments: dict):
        """
        Initialization involves preparing the algorithm and environment for training.
        If loading and continuing training from a directory, everything will also be set up here.

        Args:
            arguments (dict): A dictionary containing all the arguments necessary for training the algorithm.
        """

        """
        Load arguments from the directory if, 
        (1) The algorithm is being rendered
        (2) The algorithm is continuing training from a directory
        """
        # flags
        self.render = arguments['render']
        self.load_from_directory = arguments['load_from_directory']
        self.directory = arguments['directory']
        # self.pomdp=arguments['pomdp']

        # loading or storing arguments
        if (self.load_from_directory is True) or (self.render is True):
            self.arguments = AlgorithmRunner.load_dictionary_from_json_file(directory=arguments['directory'], filename='arguments.json')
        else:
            self.arguments = arguments

        # since the loaded arguments might have different flag values from a previous run
        self.arguments['render'] = self.render
        self.arguments['load_from_directory'] = self.load_from_directory
        self.arguments['directory'] = self.directory

        """ Save arguments to a file (if not rendering or loading from the directory) """
        if (not self.load_from_directory) and (not self.render):
            AlgorithmRunner.save_dictionary_to_json_file(
                directory=self.arguments['directory'],
                filename='arguments.json',
                dictionary=self.arguments
            )

        """
        If loading and continuing training from a directory,
        load the previous environment context,
        else create a new environment context.
        
        (Does not affect rendering)
        """
        if self.load_from_directory is True:
            self.env_context = torch.load(f=os.path.join(self.arguments['directory'], 'checkpoint/', 'env_context.pt'))
        else:
            env = create_environment(env_name=self.arguments['env_name'])
            # env = create_environment(env_name=self.arguments['env_name'], pomdp=self.pomdp)
            self.env_context = {
                'env': env,
                'time_step_counter': 0,
                'episode_counter': 0,
                'number_of_agents': env.max_num_agents,
                'evaluation_scores': [DynamicArray(size=0, dtype=float) for _ in range(env.max_num_agents)],
                'evaluation_score_sums': DynamicArray(size=0, dtype=float),
                'evaluation_time_steps': DynamicArray(size=0, dtype=int),
                'actions_list': [],
                'rewards_list': [],
                'observations_list': [],
                'truncations_list': [],
                'terminations_list': [],
                'done': [False] * env.max_num_agents
            }

        """
        Create the algorithm.
        
        Load the trained algorithm from the checkpoint directory if,
        (1) Continuing training from the directory
        (2) Rendering the algorithm
        """
        self.algorithm = create_algorithm(
            algorithm_name=self.arguments['algorithm_name'],
            env_name=self.arguments['env_name'],
            arguments=self.arguments,
            pomdp=self.arguments['pomdp'],
            pomdp_type=self.arguments['pomdp_type']
        )
        if self.load_from_directory is True:
            self.arguments['save_replay_buffer'] = True
            self.algorithm.load_checkpoint(
                directory=self.arguments['directory'],
                load_replay_buffer=True
            )
        elif self.render is True:
            self.arguments['save_replay_buffer'] = False
            self.algorithm.load_checkpoint(
                directory=self.arguments['directory'],
                load_replay_buffer=False
            )

        """ the data manager has all the utility functions for exporting data as plots and .csv files """
        self.data_manager = DataManager()

    def print_arguments(self):
        """
        A function for printing out arguments as a sanity check and for debugging.
        Useful at the start of training.
        """
        print()
        print('--- Arguments ---')
        for key, value in self.arguments.items():
            print(f'{key}: {value}')
        print()

    def print_basic_info(self):
        """
        Prints basic info from the .json file in the checkpoint directory.
        Useful when continuing training.
        """
        basic_info: dict = AlgorithmRunner.load_dictionary_from_json_file(
            directory=os.path.join(self.arguments['directory'], 'checkpoint/'),
            filename='basic_info.json'
        )

        print()
        print('--- Basic Info ---')
        for key, value in basic_info.items():
            print(f'{key}: {value}')
        print()

    def print_env_context(self):
        """
        Prints information about the environment context.
        Useful for a basic sanity check when starting training or continuing training from a directory.
        """
        excluded_keys = ['observations_list', 'actions_list', 'rewards_list', 'next_observations_list']
        print()
        print('--- Environment Context ---')
        for key, value in self.env_context.items():
            if key == 'evaluation_scores':
                string_list = [score.size for score in value]
                print(f'{key}: size: {string_list}')
            elif key == 'evaluation_score_sums':
                print(f'{key}: size: {value.size}')
            elif key == 'evaluation_time_steps':
                print(f'{key}: size: {value.size}')
            elif key not in excluded_keys:
                print(f'{key}: {value}')
            else:
                pass
        print()

    @staticmethod
    def save_dictionary_to_json_file(
            directory: str,
            filename: str,
            dictionary: dict
    ):
        """
        Save a Python dictionary to a file as human-readable JSON.

        Args:
            directory (str): The directory to save the file to. Example >>> "./folder/path/"

            filename (str): The name to give the file. Example >>> "human_readable_json.txt" or "filename.json"

            dictionary (dict): The Python dictionary to save as human-readable JSON.
        """
        # make sure the directory exists before trying to save the arguments to a file
        if not os.path.exists(directory):
            os.makedirs(directory)

        # save the arguments to a file as human-readable JSON
        filepath = os.path.join(directory, filename)
        with open(filepath, "w") as file:
            json.dump(dictionary, file, indent=2)

    @staticmethod
    def load_dictionary_from_json_file(
            directory: str,
            filename: str
    ):
        """
        Load a Python dictionary from a human-readable JSON file.

        Args:
            directory (str): The directory to load the file from. Example >>> "./folder/path/"

            filename (str): The name of the file to load. Example >>> "human_readable_json.txt" or "filename.json"
        """
        # read the json file
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            json_data_as_string = file.read()

        # parse the string for JSON and create a dictionary
        dictionary: dict = json.loads(json_data_as_string)

        return dictionary

    @staticmethod
    def numpy_to_tensor_list(numpy_list: list[np.ndarray], dtype=torch.float) -> list[torch.Tensor]:
        """ Converts a list of numpy arrays into a list of torch tensors, and returns the tensor list. """
        return [torch.tensor(numpy_array, dtype=dtype) for numpy_array in numpy_list]

    def save_checkpoint(self):
        """
        Save all the necessary information for continuing the training to the checkpoint directory.

        Saves,
        1. Algorithm (with trained weights and replay buffer)
        2. Environment Context (to begin training where it left off at the checkpoint)
        3. Basic Information
        """
        print(f'--- Saving Checkpoint [{self.env_context["time_step_counter"]}] ---')

        # basic info
        basic_info = {
            'env_name': self.arguments['env_name'],
            'time step': self.env_context['time_step_counter'],
            'episode': self.env_context['episode_counter']
        }

        # save basic info
        AlgorithmRunner.save_dictionary_to_json_file(
            directory=os.path.join(self.arguments['directory'], 'checkpoint'),
            filename='basic_info.json',
            dictionary=basic_info
        )

        # save environment context
        torch.save(
            obj=self.env_context,
            f=os.path.join(self.arguments['directory'], 'checkpoint', 'env_context.pt')
        )

        # save agent network weights and replay buffer
        self.algorithm.save_checkpoint(
            directory=self.arguments['directory'],
            save_replay_buffer=self.arguments['save_replay_buffer']
        )

    def export_plot_and_csv(self):
        """
        Exports data as a plot and .csv file to the directory.
        """
        print(f'--- Exporting Plot and CSV [{self.env_context["time_step_counter"]}] ---')

        # get the necessary data as local variables for convenience
        directory = self.arguments['directory']
        env_name = self.arguments['env_name']
        env = self.env_context['env']
        number_of_agents = len(env.possible_agents)
        time_step_counter = self.env_context['time_step_counter']
        evaluation_scores = self.env_context['evaluation_scores']
        evaluation_score_sums = self.env_context['evaluation_score_sums']
        evaluation_time_steps = self.env_context['evaluation_time_steps']

        # get a list of all the agent names
        legend_labels = deepcopy(env.possible_agents)
        # add one more series label for the summed score
        legend_labels.append('Score Sum')

        # create the list of x_series values
        x_series = [evaluation_time_steps[:] for _ in range(number_of_agents + 1)]  # +1 for the sum series

        # create the list of y_series values
        y_series = [evaluation_scores[i][:] for i in range(number_of_agents)]
        y_series.append(evaluation_score_sums[:])

        # create the data directory path
        data_directory_path = os.path.join(directory, 'data/')

        # export the plot
        self.data_manager.export_plot_multiple_series(
            directory=data_directory_path,
            filename=f'plot_time_step_{time_step_counter}.png',
            title=f'Environment {env_name} Performance',
            x_label='Time Step',
            y_label='Score',
            legend_labels=legend_labels,
            x_series=x_series,
            y_series=y_series,
            show_grid_lines=True,
            calculate_moving_average=False,
            moving_average_window_size=25
        )

        # create a dictionary with the evaluation scores and export to a .csv file
        data_dictionary = {}
        data_dictionary['Time Step'] = evaluation_time_steps[:]
        for index, agent_name in enumerate(deepcopy(env.possible_agents)):
            key = agent_name + ' Score'
            data_dictionary[key] = evaluation_scores[index][:]
        data_dictionary['Score Sum'] = evaluation_score_sums[:]

        # export the .csv file
        self.data_manager.export_csv_from_dictionary(
            directory=data_directory_path,
            filename=f'data_time_step_{time_step_counter}.csv',
            dictionary=data_dictionary
        )

    def run_episode(self):
        """
        Runs an episode.
        """

        # begin a new environment time step
        # while the current episode is not truncated or terminated (no agent has truncated/terminated)
        while not any(self.env_context['done']):
            # each agent chooses an action given their current states in parallel
            # at the beginning of trials, agents take random actions for state space exploration
            if self.env_context['time_step_counter'] < self.arguments['start_steps']:
                # random actions sampled from the agents' action spaces
                actions = {
                    agent: self.env_context['env'].action_space(agent).sample() for agent in self.env_context['env'].agents
                }
            else:
                # actions from the agents' policies
                if self.arguments['algorithm_name'] == 'ma_lstm_td3':
                    # ma_lstm_td3 expects a list of np.ndarray
                    actions = self.algorithm.choose_action(
                        observations=self.env_context['observations_list'],
                        evaluate=False
                    )
                else:
                    # ma_ddpg, ma_td3, expect a list of pytorch tensors
                    actions = self.algorithm.choose_action(
                        observations=AlgorithmRunner.numpy_to_tensor_list(self.env_context['observations_list']),
                        evaluate=False
                    )

            # env.step() returns information as dictionaries with the keys being the agents' IDs
            next_observations, rewards, terminations, truncations, info = self.env_context['env'].step(actions)

            # get the values of each dictionary and convert them into lists
            self.env_context['actions_list']: list[np.ndarray] = list(actions.values())
            self.env_context['rewards_list']: list[np.ndarray] = list(rewards.values())
            self.env_context['next_observations_list']: list[np.ndarray] = list(next_observations.values())
            self.env_context['truncations_list']: list[np.ndarray] = list(truncations.values())
            self.env_context['terminations_list']: list[np.ndarray] = list(terminations.values())

            # set the done flags (if the agents have truncated or terminated, the episode ends)
            self.env_context['done'] = [
                truncated or terminated
                for truncated, terminated in zip(self.env_context['truncations_list'], self.env_context['terminations_list'])
            ]

            self.algorithm.store_experience(
                observations=self.env_context['observations_list'],
                actions=self.env_context['actions_list'],
                rewards=self.env_context['rewards_list'],
                next_observations=self.env_context['next_observations_list'],
                terminations=self.env_context['terminations_list']
            )

            # set the observations for the next time step
            self.env_context['observations_list'] = self.env_context['next_observations_list']

            # increment the time step counter, since another step finished
            self.env_context['time_step_counter'] += 1

            # check if the algorithm should perform an optimization step
            if (self.env_context['time_step_counter'] % self.arguments['update_every'] == 0) and (
                    self.env_context['time_step_counter'] >= self.arguments['update_after']):
                # perform a specified number of optimization steps for the current time step
                for _ in range(self.arguments['optimization_steps']):
                    self.algorithm.learn()

            """ evaluate the algorithms performance at intervals """
            if self.env_context['time_step_counter'] % self.arguments['evaluation_interval'] == 0:
                # evaluate the performance of the algorithm
                agent_scores = self.evaluate_algorithm()

                # record the performance of the agents and algorithm
                for index, agent_score in enumerate(agent_scores):
                    self.env_context['evaluation_scores'][index].append(agent_score)
                self.env_context['evaluation_score_sums'].append(agent_scores.sum())
                self.env_context['evaluation_time_steps'].append(self.env_context['time_step_counter'])

                # print performance to terminal
                print(
                    f'| Episode | {self.env_context["episode_counter"] + 1} '
                    f'| Time Step | {self.env_context["time_step_counter"]} '
                    f'| Score | {agent_scores} '
                    f'| Summed Score | {agent_scores.sum()} |'
                )

            """ export data to a directory at intervals """
            if self.env_context['time_step_counter'] % self.arguments['log_interval'] == 0:
                self.export_plot_and_csv()

            """ save the parameters to a directory at intervals """
            if self.env_context['time_step_counter'] % self.arguments['checkpoint_interval'] == 0:
                # don't waste time saving checkpoints until after the random agent actions (start steps) finishes
                if self.env_context['time_step_counter'] >= self.arguments['start_steps']:
                    self.save_checkpoint()

    def evaluate_algorithm(self) -> np.ndarray:
        """
        The multi-agent algorithm's performance is evaluated.
        During evaluation, no noise is added to the agents' actions.
        Noise is only added during training for state space exploration.

        Returns:
            Returns a np.ndarray of scores for each agent.
            Example: 3 agents will return a numpy array of shape (3) with each of their average scores for the episodes
        """
        # a deepcopy of the environment is created so that the original environment,
        # which is still being used for training,
        # is unaffected
        evaluation_env = deepcopy(self.env_context['env'])
        _, _ = evaluation_env.reset()
        number_of_agents: int = evaluation_env.max_num_agents

        # the number of evaluation episodes to run and storage for the agents' scores
        evaluation_episodes: int = self.arguments['evaluation_episodes']
        evaluation_scores = np.zeros([evaluation_episodes, number_of_agents], dtype=float)

        # loop over the episodes and test the algorithm
        for i in range(evaluation_episodes):
            # reset the environment for a new episode
            observations_dictionary, _ = evaluation_env.reset()

            # list of observations for each agent (conversion from dictionary)
            observations_list = list(observations_dictionary.values())

            # list of flags indicating if the episode has terminated (any agent has terminated)
            done = [False] * number_of_agents

            # score for each agent of the current episode
            reward_sum = np.zeros([number_of_agents], dtype=float)

            if self.arguments['algorithm_name'] == 'ma_lstm_td3':
                self.algorithm.episode_reset(starting_observations=deepcopy(observations_list))

            while not any(done):
                # each agent chooses an action given their current states in parallel
                # evaluate=True because there is no noise added to actions during evaluation
                if self.arguments['algorithm_name'] == 'ma_lstm_td3':
                    # ma_lstm_td3 expects a list of np.ndarray
                    actions = self.algorithm.choose_action(
                        observations=observations_list,
                        evaluate=True
                    )
                else:
                    # ma_ddpg, ma_td3, expect a list of pytorch tensors
                    actions = self.algorithm.choose_action(
                        observations=AlgorithmRunner.numpy_to_tensor_list(observations_list),
                        evaluate=True
                    )

                # evaluation_env.step() returns information as dictionaries with the keys being the agents' IDs
                next_observations, rewards, terminations, truncations, info = evaluation_env.step(actions)

                # get the values of each dictionary and convert them into lists
                rewards_list = list(rewards.values())
                next_observations_list = list(next_observations.values())
                truncations_list = list(truncations.values())
                terminations_list = list(terminations.values())

                # set the done flags (if the agents have truncated or terminated, the episode ends)
                done = [truncated or terminated for truncated, terminated in zip(truncations_list, terminations_list)]

                # set the observations for the next time step
                observations_list = next_observations_list

                # keep track of the score
                reward_sum += np.array(rewards_list)

            # add the episode score to the list of scores
            evaluation_scores[i] = reward_sum

        # find the average score, for each agent, for all the evaluation episodes
        average_score = np.mean(evaluation_scores, axis=0)

        return average_score

    def train_algorithm(self):
        """
        """

        """ print out some important information before the training begins (sanity check) """
        self.print_arguments()
        if self.load_from_directory is True:
            self.print_basic_info()
        self.print_env_context()
        print()
        print('--- Replay Buffer ---')
        print(f'memory_counter: {self.algorithm.ma_replay_buffer.memory_counter}')
        print()

        """ record the initial performance of the untrained algorithm """
        if self.load_from_directory is False:
            agent_scores: np.ndarray = self.evaluate_algorithm()
            for index, agent_score in enumerate(agent_scores):
                self.env_context['evaluation_scores'][index].append(agent_score)
            self.env_context['evaluation_score_sums'].append(agent_scores.sum())
            self.env_context['evaluation_time_steps'].append(0)

        """ run the environment episode loop for training """
        # counters for the time steps and episodes elapsed
        if self.load_from_directory is False:
            self.env_context['time_step_counter'] = 0
            self.env_context['episode_counter'] = 0

        """ begin a new episode """
        while self.env_context['time_step_counter'] < self.arguments['training_steps']:

            # if not loading from a directory,
            # then we can reset the environment here for the new episode
            if self.load_from_directory is False:
                # reset the environment for a new episode
                observations_dictionary, _ = self.env_context['env'].reset()

                # list of observations for each agent (conversion from dictionary)
                self.env_context['observations_list'] = list(observations_dictionary.values())

                # list of flags indicating if the episode has terminated
                self.env_context['done'] = [False] * self.env_context['number_of_agents']

                # reset observation
                if self.arguments['algorithm_name'] == 'ma_lstm_td3':
                    self.algorithm.episode_reset(starting_observations=deepcopy(self.env_context['observations_list']))

            """ run an episode """
            self.run_episode()

            """ the current episode has just finished here """
            self.env_context['episode_counter'] += 1

            if (self.env_context['episode_counter'] % 5 == 0) and (self.env_context['time_step_counter'] % self.arguments['evaluation_interval'] != 0):
                print(f'--- Episode {self.env_context["episode_counter"]} ---')

            # if loading from a directory,
            # we want to continue running the episode first run_episode() because it might have been in progress
            # then we can reset for a new episode after it has terminated
            if self.load_from_directory is True:
                # reset the environment for a new episode
                observations_dictionary, _ = self.env_context['env'].reset()

                # list of observations for each agent (conversion from dictionary)
                self.env_context['observations_list'] = list(observations_dictionary.values())

                # list of flags indicating if the episode has terminated
                self.env_context['done'] = [False] * self.env_context['number_of_agents']

                if self.arguments['algorithm_name'] == 'ma_lstm_td3':
                    self.algorithm.episode_reset(starting_observations=deepcopy(self.env_context['observations_list']))

    def render_algorithm(self):
        """ Calling this function will render the algorithm in the environment for visualization. """

        """ create the environment for rendering """
        env = create_environment(env_name=self.arguments['env_name'], render=True)
        _, _ = env.reset()
        number_of_agents = len(env.possible_agents)

        """ run the environment loop """
        max_time_steps = 100_000  # render the algorithm for this many time steps

        # counters for the time steps and episodes elapsed
        time_step_counter = 0
        episode_counter = 0

        while time_step_counter < max_time_steps:
            # reset the environment for a new episode
            observations, _ = env.reset()
            observations = list(
                observations.values())  # list of observations for each agent (conversion from dictionary)
            done = [False] * number_of_agents  # list of flags indicating if the episode has terminated
            agent_scores = np.zeros(shape=[number_of_agents], dtype=float)
            score_sum = 0.0

            while not any(done):
                # render the environment
                pygame.event.pump()  # prevents animation from freezing (on my computer, PettingZoo rendering uses pygame)
                time.sleep(0.08)  # time delay to slow down animation between time steps

                # each agent chooses an action given their current states in parallel
                actions = self.algorithm.choose_action(AlgorithmRunner.numpy_to_tensor_list(observations), evaluate=True)

                # parallel_env.step() returns information as dictionaries with the keys being the agents' IDs
                next_observations, rewards, terminations, truncations, info = env.step(actions)

                # get the values of each dictionary and convert them into lists
                next_observations_list = list(next_observations.values())
                truncations_list = list(truncations.values())
                terminations_list = list(terminations.values())
                rewards_list = list(rewards.values())

                # add the rewards for each agent to their episode scores
                agent_scores += np.array(rewards_list)
                score_sum += np.array(rewards_list).sum()

                # set the done flags (if the agents have truncated or terminated, the episode ends)
                done = [truncated or terminated for truncated, terminated in zip(truncations_list, terminations_list)]

                # set the observations for the next time step
                observations = next_observations_list

                # increment the time step counter, since another step finished
                time_step_counter += 1

            # print performance
            print(f'Episode | {episode_counter + 1} | Time Step | {time_step_counter} | Agent Scores | {agent_scores} | Score Sum | {score_sum}')

            # another episode has finished
            episode_counter += 1


if __name__ == '__main__':
    pass
