"""
Description:
    A very quickly written script that will load several trained multi-agent reinforcement learning algorithms.
    These different trained algorithms will independently choose actions in the same environment.
    Then their actions will be averaged and used in the environment.

    The idea is that by using an ensemble of MARL algorithms, actions will be more robust.
    The performance will be roughly evaluated and averaged for several episodes.
    Then the algorithm and environment will be rendered.
"""
import os
import json
import time
import pygame
from copy import deepcopy
import numpy as np
import torch
from source.environment.create_environment import create_environment
from source.multi_agent_algorithm.create_algorithm import create_algorithm


def numpy_to_tensor_list(numpy_list: list[np.ndarray], dtype=torch.float) -> list[torch.Tensor]:
    """ Converts a list of numpy arrays into a list of torch tensors, and returns the tensor list. """
    return [torch.tensor(numpy_array, dtype=dtype) for numpy_array in numpy_list]


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


def run_environment(algorithms, use_ensemble: bool, agent_index: int, env_name: str, number_of_episodes: int, render: bool):
    """
    Args:
        algorithms: A list of the trained algorithms which will run together as an ensemble.
        use_ensemble (bool): If set to true, actions from all algorithms will be averaged and used, else, just a single agent's actions (agent_index) will be used.
        agent_index (int): If not using an ensemble, then just this agent's actions will be used.
        env_name (str): The name of the environment to create and run.
        number_of_episodes: The number of episodes to run for.
        render: If set to true, the algorithm and environment will be rendered.

    Returns:
        A numpy array containing the average scores for each agent over all episodes,
        and the average sum of the agents' scores over all episodes.
    """
    evaluation_env = create_environment(env_name=env_name, render=render)
    _, _ = evaluation_env.reset()
    number_of_agents: int = evaluation_env.max_num_agents

    # storage for the agents' scores
    evaluation_scores = np.zeros([number_of_episodes, number_of_agents], dtype=float)

    # loop over the episodes and test the algorithm
    for i in range(number_of_episodes):
        # reset the environment for a new episode
        observations_dictionary, _ = evaluation_env.reset()

        # list of observations for each agent (conversion from dictionary)
        observations_list = list(observations_dictionary.values())

        # list of flags indicating if the episode has terminated (any agent has terminated)
        done = [False] * number_of_agents

        # score for each agent of the current episode
        reward_sum = np.zeros([number_of_agents], dtype=float)

        while not any(done):
            # if rendering the environment
            if render is True:
                pygame.event.pump()  # prevents animation from freezing (on my computer, PettingZoo rendering uses pygame)
                time.sleep(0.08)  # time delay to slow down animation between time steps

            # if using an ensemble: the actions are the average of all algorithms
            # not using an ensemble: the actions are just from one specified agent
            if use_ensemble is True:
                actions: dict = {}
                # each algorithm has its agents choose actions given the observation
                all_algorithm_actions: list[dict] = [
                    algorithm.choose_action(numpy_to_tensor_list(observations_list), evaluate=True)
                    for algorithm in algorithms
                ]

                # average the actions (sum then divide)
                for index, dictionary in enumerate(all_algorithm_actions):
                    # sum
                    if index == 0:
                        actions = deepcopy(dictionary)
                    else:
                        for key in dictionary.keys():
                            actions[key] += dictionary[key]
                    # divide
                    for key, value in actions.items():
                        actions[key] = value / len(algorithms)
            else:
                actions: dict = algorithms[agent_index].choose_action(numpy_to_tensor_list(observations_list), evaluate=True)

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

        # if rendering print out every episode score
        if render is True:
            print(
                f'| Episode | {i+1} '
                f'| Agent Scores | {evaluation_scores[i]} '
                f'| Score Sum | {evaluation_scores[i].sum()}'
            )
        else:
            if (i+1) % 10 == 0:
                pass
                """print(
                    f'| Episode | {i + 1} '
                    f'| Agent Scores | {evaluation_scores[i]} '
                    f'| Score Sum | {evaluation_scores[i].sum()}'
                )"""

    # find the average score, for each agent, for all the evaluation episodes
    average_score = np.mean(evaluation_scores, axis=0)

    return average_score, evaluation_scores


def main():
    # the directories with the trained algorithms to load and create an ensemble
    directories: list[str] = [
        './tests/test_simple_spread_v3_td3',
        './tests/test_simple_spread_v3_td3_2',
        './tests/test_simple_spread_v3_td3_3'
    ]

    # load the arguments for each of the algorithms
    arguments: list[dict] = [
        load_dictionary_from_json_file(directory=directory, filename='arguments.json')
        for directory in directories
    ]

    # greatly reduce the replay buffer sizes (my computer doesn't have a massive amount of RAM)
    # the replay buffer is not even necessary since the algorithms are being evaluated and not trained
    for args in arguments:
        args['buffer_size'] = 1024

    # create the three algorithms that will be running as an ensemble
    algorithms = [
        create_algorithm(algorithm_name=args['algorithm_name'], env_name=args['env_name'], arguments=args)
        for args in arguments
    ]

    # lol,
    # I forgot to load the trained weights
    # no wonder my agents and ensemble was horrible
    agent_names = create_environment(env_name=arguments[0]['env_name'], render=False).possible_agents
    for i in range(len(algorithms)):
        algorithms[i].load_checkpoint(
            directory=directories[i],
            agent_names=agent_names,
            load_replay_buffer=False
        )

    # check the average performance of each individual trained algorithm
    print()
    number_of_episodes = 500
    algorithm_std = []
    for index, algorithm in enumerate(algorithms):
        average_score, evaluation_scores = run_environment(
            algorithms=algorithms,
            use_ensemble=False,
            agent_index=index,
            env_name=arguments[0]['env_name'],
            number_of_episodes=number_of_episodes,
            render=False
        )
        algorithm_std.append(evaluation_scores.sum(axis=1).std())
        print(
            f'| Agent Index | {index} '
            f'| Number of Episodes | {number_of_episodes} '
            f'| Average Agent Scores | {average_score} '
            f'| Average Score Sum | {average_score.sum()}'
            f'| Directory | {directories[index]}'
        )

    # check the performance of the ensemble
    average_score, evaluation_scores = run_environment(
        algorithms=algorithms,
        use_ensemble=True,  # agent_index value doesn't matter when use_ensemble is set to True
        agent_index=0,
        env_name=arguments[0]['env_name'],
        number_of_episodes=number_of_episodes,
        render=False
    )
    ensemble_std = evaluation_scores.sum(axis=1).std()
    print(
        f'| Ensemble | YES '
        f'| Number of Episodes | {number_of_episodes} '
        f'| Average Agent Scores | {average_score} '
        f'| Average Score Sum | {average_score.sum()}'
    )
    print()

    # I thought it would be interesting to compare the standard deviations
    # it seems that the average score of the ensemble is not better than the individual algorithms,
    # but perhaps the ensemble would show more consistency (i.e. smaller standard deviation)
    #
    # 'hoping' there is some kind of benefit to averaging the algorithms output
    for i in range(len(algorithms)):
        print(f'| Agent Index | {i} | STD | {algorithm_std[i]}')
    print(f'| Ensemble | YES | STD | {ensemble_std}')
    print()

    # second, run the environment with rendering to visually see what the ensemble performance looks like
    run_environment(
        algorithms=algorithms,
        env_name=arguments[0]['env_name'],
        use_ensemble=True,
        agent_index=0,
        number_of_episodes=number_of_episodes,
        render=True
    )


if __name__ == '__main__':
    main()
