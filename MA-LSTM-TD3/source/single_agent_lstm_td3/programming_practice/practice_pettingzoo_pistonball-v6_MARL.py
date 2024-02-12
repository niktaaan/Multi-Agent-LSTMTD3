"""
Description:
    This is a practice program where I am trying to see how to use PettingZoo for Multi-agent Reinforcement Learning.
    PettingZoo is a framework with many different MARL environments.

    In the program below I will try to have multiple agents operate in a MARL environment.

    I tried to create some agents, but then I realized that the state information are rather large pictures.
    Thus, there is not enough space on the GPU to store thousands of experiences in the replay buffer.
    The observations would need some pre-processing step before being used and stored.

    for the piston environment,
    it is possible for many of the agents to share training data and policies
    the pistons that have walls on the side are different
    I will just have all different rl agents learning their own decentralized policies and training

Author:
    Jordan Cramer

Date:
    2023-08-09
"""
from pettingzoo.butterfly import pistonball_v6
import numpy as np


if __name__ == '__main__':
    """ initialize a pettingzoo environment """
    # env = pistonball_v6.env(continuous=True, render_mode='human')
    env = pistonball_v6.env(continuous=True)
    env.reset()

    # checking the agents of the environment
    print('env agents:', env.agents)  # ['piston_0', 'piston_1', ... , 'piston_19']

    # checking the number of agents
    number_of_agents = len(env.agents)
    print('number of agents:', number_of_agents)

    # checking the action and observation spaces of an agent
    print('action space', env.action_space(env.agents[0]))  # Box(-1.0, 1.0, (1,), float32)
    print('observation space:', env.observation_space(env.agents[0]))  # Box(0, 255, (457, 120, 3), uint8)

    # checking if all the agents have the same shape observation space, or if their observation shape are different
    all_same_shape = True
    os = env.observation_space(env.agents[0])
    for agent in env.agents:
        os2 = env.observation_space(agent)
        if os != os2:
            all_same_shape = False
            break
        os = os2
    print('Are all observation shapes the same?', all_same_shape)  # True (They are all the same shape apparently)

    """ create different rl agents for each of the pistons """
    # I tried to create some agents, but then I realized that the state information are rather large pictures.
    # Thus, there is not enough space on the GPU to store thousands of experiences in the replay buffer.
    # The observations would need some pre-processing step before being used and stored.
    #
    # for the piston environment,
    # it is possible for many of the agents to share training data and policies
    # the pistons that have walls on the side are different
    # I will just have all different rl agents learning their own decentralized policies and training
    """td3_agents = []
    observation_space_shape = [457, 120, 3]
    for i in range(number_of_agents):
        td3_agents.append(
            TwinDelayedAgent(
                alpha=0.001,
                beta=0.001,
                input_dims=observation_space_shape,
                tau=0.005,
                batch_size=100,
                max_size=1_000,
                layer1_size=400,
                layer2_size=300,
                warmup=10,
                n_actions=1,
                checkpoint_dir=f'./checkpoint_agent_{i}',
                min_action=-1.0,
                max_action=1.0
            )
        )"""

    """ train over several episodes """
    number_of_episodes = 1000
    for episode in range(number_of_episodes):
        env.reset()
        done = False  # termination or truncation ends the episode
        returns = np.zeros(number_of_agents, dtype=float)

        while not done:
            """ iterate through the different agents and perform actions """
            for i, agent in enumerate(env.agent_iter()):
                # either sample from the action space or insert your own policy for agent actions
                action = env.action_space(agent).sample()

                # tell the agent to perform the action
                env.step(action)

                # get the values for the agent
                observation, reward, termination, truncation, info = env.last()

                # record the rewards
                returns[i % number_of_agents] += reward

                # interesting information
                #
                # printing out information about the values
                # print('observation type:', type(observation))  # observation type: <class 'numpy.ndarray'>
                # print('observation shape:', observation.shape)  # (457, 120, 3)  (this is 3 colour channels I think)
                # print('reward type:', type(reward))  # <class 'float'>
                # print('reward:', reward)  # example: -0.1
                # print('termination type:', type(termination))  # <class 'bool'>
                # print('termination:', termination)  # example: False
                # print('truncation type:', type(truncation))  # <class 'bool'>
                # print('truncation', truncation)  # example: True
                # print(info)  # {}

                # check if the episode is done (if done, break from the episode loop)
                done = termination or truncation
                if done:
                    break

                # render the environment if the env render_mode == 'human'
                # env.render()

        """ episode end performance """
        print(f'Episode {episode+1} Total Return: {returns.sum():.3f}')
        # print(f'Agent Returns:', returns)
