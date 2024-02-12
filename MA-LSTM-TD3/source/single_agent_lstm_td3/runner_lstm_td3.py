"""
Description:
    Train a LSTM-TD3 agent for MDP and POMDP environments.

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
import random
import torch
import numpy as np
from source.single_agent_lstm_td3.algorithm.agent_lstm_td3 import LSTMTD3
from source.single_agent_lstm_td3.environment.mdp_wrapper import MDPWrapper
from source.single_agent_lstm_td3.environment.pomdp_wrapper import POMDPWrapper
from source.utility.data_manager import DataManager


def train(
        env,
        agent: LSTMTD3,
        start_steps: int,
        update_after: int,
        update_every: int,
        number_of_episodes: int,
        save_directory: str = None,
        save_interval: int = 50
):
    """
    Train the reinforcement agent.
    A reinforcement agent will learn a given environment.

    Args:
        env: The environment that the agent will train for.
        agent: The reinforcement learning agent.
        start_steps (int): The number of starting time steps where the agents will take random actions instead of using their policy. This helps initial state space exploration.
        update_after (int): The number of env interactions to collect before starting to do gradient descent updates. Ensures the replay buffer is full enough for useful updates. Should not be less than the batch size.
        update_every (int): The number of env interactions (time steps) that should elapse between gradient descent updates.
        number_of_episodes (int): The number of episodes to run the environment.
        save_directory (str): Performance and trial information will be saved to this directory.
        save_interval (int): Every save_interval number of episodes, save performance data to the save_directory.
    """

    # prepare to export performance plots and data
    data_manager = DataManager()

    # time step counter for counting total number of elapsed time steps
    time_step_counter = 0

    """ loop over environment episodes """
    for episode in range(number_of_episodes):

        # reset the environment for a new episode
        observation, info = env.reset()
        episode_return = 0  # episode return
        episode_length = 0  # episode time step length

        # IMPORTANT
        # prepare history buffers to store observation-action histories of the correct length and size
        agent.initialize_history_buffer(starting_observation=observation)

        """ loop over environment time steps """
        done = False  # flag for checking if the episode has truncated or terminated
        while not done:
            # Until start_steps have elapsed, randomly sample actions from a uniform distribution for better exploration.
            # Afterward, use the learned policy (with some noise, via act_noise).
            if time_step_counter > start_steps:
                # the agent chooses an action
                action = agent.choose_action(
                    observation=observation,  # current environment observation
                    add_noise=True
                )
            else:
                # randomly sample an action
                action = env.action_space.sample()

            # agent performs the action
            next_observation, reward, truncation, termination, _ = env.step(action)

            # check if the episode is done
            done = truncation or termination

            # add the time step reward to the current episode return
            # increment the time step counter
            episode_return += reward
            episode_length += 1
            time_step_counter += 1

            # store experience to replay buffer
            agent.replay_buffer.store_experience(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                termination=termination
            )

            # IMPORTANT
            # must add the observation-action pair each time step for the histories to be correct
            # add the most recent observation and action to the current history
            agent.add_observation_action_to_history_buffer(
                observation=observation,
                action=action
            )

            # update the most recent observation
            observation = next_observation

            # learning step
            if time_step_counter >= update_after and time_step_counter % update_every == 0:
                # example: if updating every 50 time steps, then it will perform 50 optimizations all at once here
                for j in range(update_every):
                    # sample a batch of experiences
                    batch = agent.replay_buffer.sample_buffer_with_history(
                        batch_size=agent.batch_size,
                        max_history_length=agent.max_history_length
                    )

                    # send all the batch values in the dictionary to the device (CPU or GPU) for calculations
                    batch = {k: v.to(agent.device) for k, v in batch.items()}

                    # perform the optimization/learning step
                    agent.learn(data=batch)

        """ episode has finished """
        # print out simple performance metrics
        print(f'Episode: {episode + 1} Return: {episode_return:.5f} Length (Time Steps): {episode_length}')

        # possibly test the deterministic performance of the agent here
        # or just use the score from the above episode where noise was being added to the agent's actions

        """ log performance to files """
        # IMPORTANT
        # record the episode score and reset for the next episode
        env.score_storage.episode_reset()

        # if it is time to save the performance (save_interval), save the performance
        if (episode + 1) % save_interval == 0:
            # export a plot of the performance to a file
            data_manager.export_plot(
                directory=save_directory,
                filename=f'Episode_{episode + 1}.png',
                scores=np.array(env.score_storage.scores, dtype=float),
                sliding_window_size=50,
                title=f'Episode {episode + 1}',
                y_label='Score',
                x_label='Episode'
            )

            # export the .csv file of the performance to a file
            data_manager.export_csv(
                directory=save_directory,
                filename=f'Episode_{episode + 1}.csv',
                scores=np.array(env.score_storage.scores, dtype=float),
                sliding_window_size=50
            )


def main():
    """
    All the trial code will be run here.

    (1) All settings are specified
        Trial Settings
        Environment Settings
        Agent Hyperparameters

    (2) The environment and agent are created

    (3) The trial is run and performance is recorded
    """

    """ trial settings """
    number_of_episodes = 1000  # alternatively, epochs with a certain number of time steps could be used
    start_steps = 10_000  # number of starting time steps where the agent will take random actions for exploration
    save_directory = './trial_results/mdp_half_cheetah/'  # save test trial results to this directory
    save_interval = 10  # every so many episodes, save performance data to the save directory

    """ environment settings """
    env_name = "HalfCheetah-v4"
    seed = random.randint(0, 1_000_000)
    use_pomdp = False
    # valid_pomdp_choices = ['remove_velocity', 'flickering', 'random_noise', 'random_sensor_missing']
    pomdp_type = 'remove_velocity'
    flicker_prob = 0.2
    random_noise_sigma = 0.1
    random_sensor_missing_prob = 0.1

    """ create the environment """
    # manually seeding pytorch and numpy pseudorandom generators
    torch.manual_seed(seed)
    np.random.seed(seed)

    # if using a POMDP use the POMDP wrapper environment, else use the MDP environment
    if use_pomdp:
        env = POMDPWrapper(
            env_name=env_name,
            pomdp_type=pomdp_type,
            flicker_prob=flicker_prob,
            random_noise_sigma=random_noise_sigma,
            random_sensor_missing_prob=random_sensor_missing_prob
        )
    else:
        env = MDPWrapper(
            env_name=env_name
        )

    """ agent settings """
    # agent variables
    observation_shape = env.observation_space.shape[0]  # the number of float values in an environment state/observation
    action_shape = env.action_space.shape[0]  # the number of action components expected from the agent
    action_min = env.action_space.low[0]  # the minimum value for an action
    action_max = env.action_space.high[0]  # the maximum value for an action
    action_noise_std = 0.1  # noise is added to agent actions for state space exploration (turned off after training)
    target_noise = 0.2  # scaling factor for target actor action noise (normal td3 algorithm)
    target_noise_clip = 0.5  # the clipping bounds (-c,+c) for the target actor action noise
    buffer_size = int(1e6)  # replay buffer size for experiences
    max_history_length = 5  # the number of observation-action pairs to include in the agent's history
    discount_factor = 0.99  # discount factor
    tau = 0.995  # polyak averaging factor for target network weight updates
    actor_learning_rate = 1e-3  # actor/policy learning rate
    critic_learning_rate = 1e-3  # critic/value learning rate
    delay_interval = 2  # the actor network will only have its weights updated every 'policy_delay' number of time steps
    batch_size = 100  # number of experiences to sample from the replay buffer during learning steps

    # flags
    use_target_policy_smoothing = True
    critic_hist_with_past_act = True
    actor_hist_with_past_act = True
    use_double_critic = True  # naturally, we want TD3 to use 2 critics to reduce Q-value overestimation bias

    # critic layer sizes
    critic_mem_pre_lstm_hid_sizes = (128,)
    critic_mem_lstm_hid_sizes = (128,)
    critic_mem_after_lstm_hid_size = (128,)
    critic_cur_feature_hid_sizes = (128,)
    critic_post_comb_hid_sizes = (128,)

    # actor layer sizes
    actor_mem_pre_lstm_hid_sizes = (128,)
    actor_mem_lstm_hid_sizes = (128,)
    actor_mem_after_lstm_hid_size = (128,)
    actor_cur_feature_hid_sizes = (128,)
    actor_post_comb_hid_sizes = (128,)

    # variables controlling the time steps when the agent performs optimization/learning
    #
    # update_after (int): Number of env interactions to collect before
    # starting to do gradient descent updates. Ensures replay buffer
    # is full enough for useful updates.
    #
    # update_every (int): Number of env interactions that should elapse
    # between gradient descent updates.
    update_after = 1000
    update_every = 50

    """ create the rl agent """
    agent: LSTMTD3 = LSTMTD3(
        observation_shape=observation_shape,
        action_shape=action_shape,
        action_min=action_min,
        action_max=action_max,
        action_noise_std=action_noise_std,
        target_noise=target_noise,
        target_noise_clip=target_noise_clip,
        buffer_size=buffer_size,
        max_history_length=max_history_length,
        discount_factor=discount_factor,
        tau=tau,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        delay_interval=delay_interval,
        batch_size=batch_size,
        critic_hist_with_past_act=critic_hist_with_past_act,
        actor_hist_with_past_act=actor_hist_with_past_act,
        use_double_critic=use_double_critic,
        use_target_policy_smoothing=use_target_policy_smoothing,
        critic_mem_pre_lstm_hid_sizes=critic_mem_pre_lstm_hid_sizes,  # (128,)
        critic_mem_lstm_hid_sizes=critic_mem_lstm_hid_sizes,  # (128,)
        critic_mem_after_lstm_hid_size=critic_mem_after_lstm_hid_size,  # (128,)
        critic_cur_feature_hid_sizes=critic_cur_feature_hid_sizes,  # (128,)
        critic_post_comb_hid_sizes=critic_post_comb_hid_sizes,  # (128,)
        actor_mem_pre_lstm_hid_sizes=actor_mem_pre_lstm_hid_sizes,  # (128,)
        actor_mem_lstm_hid_sizes=actor_mem_lstm_hid_sizes,  # (128,)
        actor_mem_after_lstm_hid_size=actor_mem_after_lstm_hid_size,  # (128,)
        actor_cur_feature_hid_sizes=actor_cur_feature_hid_sizes,  # (128,)
        actor_post_comb_hid_sizes=actor_post_comb_hid_sizes,  # (128,)
    )

    """ run test trial """
    train(
        env=env,
        agent=agent,
        start_steps=start_steps,
        update_after=update_after,
        update_every=update_every,
        number_of_episodes=number_of_episodes,
        save_directory=save_directory,
        save_interval=save_interval
    )


if __name__ == '__main__':
    main()
