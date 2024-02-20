from source.environment.create_environment import create_environment
from source.multi_agent_algorithm.ma_ddpg import MADDPG
from source.multi_agent_algorithm.ma_td3 import MATD3
from source.multi_agent_algorithm.ma_lstm_td3 import MALSTMTD3
from source.environment.environment_wrapper import Environment_Wrapper

ef create_algorithm(
        algorithm_name: str,
        env_name: str,
        arguments: dict,
        pomdp: bool(),
        pomdp_type: str
):
    """

    Args:
        algorithm_name (str): The name of the algorithm to create. "ma_ddpg" "ma_td3" "ma_lstm_td3"

        env_name (str): The name of the PettingZoo environment that the algorithm will be run for.

        arguments (dict): A dictionary containing all the necessary arguments and hyperparameters to create the algorithm.

    Returns:
        The appropriate multi-agent algorithm to run for the environment.
    """
    env = create_environment(env_name=env_name, render=False)
    env.reset()
    if pomdp is True:
       env=Environment_Wrapper(env, pomdp_type)
       env.reset()
    else:
       env.reset()

    # get the number of agents in the environment
    number_of_agents = env.max_num_agents

    # create a list with the observation/input sizes of the agents
    observation_sizes = []
    for agent in env.agents:
        observation_sizes.append(env.observation_space(agent).shape[0])

    # create a list with the action sizes of the agents
    action_sizes = []
    for agent in env.agents:
        action_sizes.append(env.action_space(agent).shape[0])

    # get the (min,max) bounds for each agents' action space
    action_space_mins = []
    action_space_maxes = []
    for agent_index in range(number_of_agents):
        # the agents might have different (min,max) bounds for actions
        agent_name = env.possible_agents[agent_index]
        action_space_mins.append(env.action_space(agent_name).low[0])
        action_space_maxes.append(env.action_space(agent_name).high[0])

    if algorithm_name == 'ma_ddpg':
        # create the agents and set up the MADDPG algorithm
        algorithm = MADDPG(
            number_of_agents=number_of_agents,
            agent_names=env.possible_agents,
            observation_sizes=observation_sizes,
            action_sizes=action_sizes,
            action_space_mins=action_space_mins,
            action_space_maxes=action_space_maxes,
            action_noise_std=arguments['action_noise_std'],
            actor_learning_rate=arguments['actor_learning_rate'],
            critic_learning_rate=arguments['critic_learning_rate'],
            actor_layer_1_size=arguments['actor_layer_1_size'],
            actor_layer_2_size=arguments['actor_layer_2_size'],
            critic_layer_1_size=arguments['critic_layer_1_size'],
            critic_layer_2_size=arguments['critic_layer_2_size'],
            buffer_size=arguments['buffer_size'],
            batch_size=arguments['batch_size'],
            discount_factor=arguments['discount_factor'],
            tau=arguments['tau']
        )
    elif algorithm_name == 'ma_td3':
        # create the agents and set up the MATD3 algorithm
        algorithm = MATD3(
            number_of_agents=number_of_agents,
            agent_names=env.possible_agents,
            observation_sizes=observation_sizes,
            action_sizes=action_sizes,
            action_space_mins=action_space_mins,
            action_space_maxes=action_space_maxes,
            action_noise_std=arguments['action_noise_std'],
            actor_learning_rate=arguments['actor_learning_rate'],
            critic_learning_rate=arguments['critic_learning_rate'],
            actor_layer_1_size=arguments['actor_layer_1_size'],
            actor_layer_2_size=arguments['actor_layer_2_size'],
            critic_layer_1_size=arguments['critic_layer_1_size'],
            critic_layer_2_size=arguments['critic_layer_2_size'],
            buffer_size=arguments['buffer_size'],
            batch_size=arguments['batch_size'],
            discount_factor=arguments['discount_factor'],
            tau=arguments['tau'],
            delay_interval=arguments['delay_interval']
        )
    elif algorithm_name == 'ma_lstm_td3':
        # create the agents and set up the MALSTMTD3 algorithm
        algorithm = MALSTMTD3(
            number_of_agents=number_of_agents,
            agent_names=env.possible_agents,
            observation_sizes=observation_sizes,
            action_sizes=action_sizes,
            action_space_mins=action_space_mins,
            action_space_maxes=action_space_maxes,
            action_noise_std=arguments['action_noise_std'],
            target_noise=arguments['target_noise'],
            target_noise_clip=arguments['target_noise_clip'],
            actor_learning_rate=arguments['actor_learning_rate'],
            critic_learning_rate=arguments['critic_learning_rate'],
            buffer_size=arguments['buffer_size'],
            batch_size=arguments['batch_size'],
            max_history_length=arguments['max_history_length'],
            discount_factor=arguments['discount_factor'],
            tau=arguments['tau'],
            delay_interval=arguments['delay_interval'],
            critic_mem_pre_lstm_hid_sizes=arguments['critic_mem_pre_lstm_hid_sizes'],
            critic_mem_lstm_hid_sizes=arguments['critic_mem_lstm_hid_sizes'],
            critic_mem_after_lstm_hid_size=arguments['critic_mem_after_lstm_hid_size'],
            critic_cur_feature_hid_sizes=arguments['critic_cur_feature_hid_sizes'],
            critic_post_comb_hid_sizes=arguments['critic_post_comb_hid_sizes'],
            actor_mem_pre_lstm_hid_sizes=arguments['actor_mem_pre_lstm_hid_sizes'],
            actor_mem_lstm_hid_sizes=arguments['actor_mem_lstm_hid_sizes'],
            actor_mem_after_lstm_hid_size=arguments['actor_mem_after_lstm_hid_size'],
            actor_cur_feature_hid_sizes=arguments['actor_cur_feature_hid_sizes'],
            actor_post_comb_hid_sizes=arguments['actor_post_comb_hid_sizes'],
            scale_lstm_gradients=arguments['scale_lstm_gradients'],
            scale_factor_lstm_gradients=arguments['scale_factor_lstm_gradients']
        )
    else:
        raise ValueError(f"Error in create_algorithm. The algorithm name {algorithm_name} was not handled.")

    return algorithm

