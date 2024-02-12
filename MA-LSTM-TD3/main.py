"""
Description:
    The purpose of this script is to have one place where all commands (command-line commands for now) can be run.
    Arguments are parsed from the command line.

    The script can train a multi-agent algorithm (MADDPG, MATD3, or MALSTMTD3),
    or render the algorithm in an environment.

    1. (MADDPG) Multi-Agent Deep Deterministic Policy Gradient
    2. (MATD3) Multi-Agent Twin Delayed Deep Deterministic Policy Gradient
    3. (MALSTMTD3) Multi-Agent Long Short-Term Memory Twin Delayed Deep Deterministic Policy Gradient

    Environments that work so far are,

    1. simple_adversary_v3
    2. simple_spread_v3
    3. simple_speaker_listener_v4

Usage:
    Training An Algorithm From The Beginning (exclude --render and exclude --load_from_directory)

    $ python main.py [arguments]

    Continuing Algorithm Training (include --load_from_directory)

    $ python main.py [arguments] --load_from_directory

    Rendering An Algorithm From Created Training Directory (include --render)

    $ python main.py [arguments] --render
    #$python main.py [arguments] --pomdp
    

Arguments:
    directory (str): The directory the algorithm will be saved to or loaded from.

    ## Optional Arguments (Flags)
    
    #pomdp(bool,false): A flag for partial observability. when this argument is included, it is set to true. The environment wrapper will generate partial observable scenarios.
    render (bool, False): A flag for rendering. When this argument is included, it is set to true. The environment will be rendered.
    load_from_directory (bool, False): A flag for loading and continuing algorithm training from the directory. When this argument is included, it is set to true.
    save_replay_buffer: A flag for making the program save the replay buffer. (Saving the replay buffer is necessary for stopping and continuing training.)

    ## Optional Arguments (Training, Evaluation, Exporting Data)

    algorithm_name (str): The name of the algorithm to train. "ma_ddpg" "ma_td3" "ma_lstm_td3"
    env_name (str): The name of the environment to train for. "simple_adversary_v3" "simple_spread_v3" "simple_speaker_listener_v4"
    training_steps (int): The total number of training steps for the trial.
    checkpoint_interval (int): The number of time steps to wait before saving the agent parameters for a checkpoint.
    log_interval (int): The number of time steps to wait before logging data to file.
    evaluation_interval (int): The number of time steps to wait before evaluating the algorithm.
    evaluation_episodes (int): When the algorithm is being evaluated, this is the number of episodes/times the algorithm is evaluated. The scores will be averaged.
    start_steps (int): The number of starting time steps where the agents will take random actions instead of using their policy. This helps initial state space exploration.
    update_after (int): The number of env interactions to collect before starting to do gradient descent updates. Ensures the replay buffer is full enough for useful updates. Should not be less than the batch size.
    update_every (int): The number of env interactions (time steps) that should elapse between gradient descent updates.
    optimization_steps (int): The number of mini-batch gradient descent operations to perform on an optimization step. It is possible to perform multiple optimization steps per time step by setting update_every and optimization_steps appropriately.

    ## Optional Arguments (Network and Algorithm Hyperparameters)

    action_noise_std (float): The standard deviation for Gaussian exploration noise added to policy actions at training time.
    actor_learning_rate (float): The learning rate for the agent actor networks.
    critic_learning_rate (float): The learning rate for the agent critic networks.
    actor_layer_1_size (int): Number of neurons in this fully connected layer.
    actor_layer_2_size (int): Number of neurons in this fully connected layer.
    critic_layer_1_size (int): Number of neurons in this fully connected layer.
    critic_layer_2_size (int): Number of neurons in this fully connected layer.
    buffer_size (int): The maximum number of experiences that can be stored in the replay buffer.
    batch_size (int): The size of a mini-batch when performing mini-batch gradient descent optimization steps.
    discount_factor (float): The discount factor.
    tau (float): A parameter controlling the proportion of the online network weights to copy over to the target networks.

    ## Optional Arguments (TD3 Relevant)

    delay_interval (int): The number of learning steps to wait before updating the actor and target networks in the TD3 algorithms.

    ## Optional Arguments (LSTM-TD3 Relevant)

    scale_lstm_gradients (bool): A flag that causes the lstm gradients to be scaled after backpropagation before optimization steps.
    scale_factor_lstm_gradients (float): The factor that the lstm gradients will be multiplied by before optimization/updates.

Example:
    Training An Algorithm From The Beginning (exclude --render and exclude --load_from_directory) ('ma_ddpg')
    
    $ python main.py "./tests/ma_ddpg_debugging/" --save_replay_buffer --algorithm_name "ma_ddpg" --env_name "simple_spread_v3" --training_steps 10000 --checkpoint_interval 2500 --log_interval 2500 --evaluation_interval 1000 --evaluation_episodes 50 --start_steps 1000 --update_after 1000 --update_every 1 --optimization_steps 1 --action_noise_std 0.1 --actor_learning_rate 1e-4 --critic_learning_rate 1e-4 --actor_layer_1_size 256 --actor_layer_2_size 256 --critic_layer_1_size 256 --critic_layer_2_size 256 --buffer_size 1_000_000 --batch_size 1024 --discount_factor 0.99 --tau 0.001
    
    Training An Algorithm From The Beginning (exclude --render and exclude --load_from_directory) ('ma_td3')

    $ python main.py "./tests/test_simple_spread_v3_td3_learn_change_performance_check/" --save_replay_buffer --algorithm_name "ma_td3" --env_name "simple_spread_v3" --training_steps 200_000 --checkpoint_interval 5_000 --log_interval 2000 --evaluation_interval 1000 --evaluation_episodes 50 --start_steps 10_000 --update_after 10_000 --update_every 1 --optimization_steps 1 --action_noise_std 0.15 --actor_learning_rate 1e-4 --critic_learning_rate 1e-4 --actor_layer_1_size 256 --actor_layer_2_size 256 --critic_layer_1_size 256 --critic_layer_2_size 256 --buffer_size 1_000_000 --batch_size 1024 --discount_factor 0.99 --tau 0.001 --delay_interval 2

    Training An Algorithm From The Beginning ('ma_lstm_td3')

    $ python main.py "./tests/test_simple_spread_v3_lstm_td3_5/" --save_replay_buffer --algorithm_name "ma_lstm_td3" --env_name "simple_spread_v3" --training_steps 1000000 --checkpoint_interval 5000 --log_interval 1000 --evaluation_interval 250 --evaluation_episodes 50 --start_steps 10000 --update_after 10000 --update_every 1 --optimization_steps 1 --action_noise_std 0.1 --actor_learning_rate 1e-3 --critic_learning_rate 1e-3 --buffer_size 1_000_000 --batch_size 512 --discount_factor 0.995 --tau 0.005 --delay_interval 2 --max_history_length 5 --critic_mem_pre_lstm_hid_sizes 128 --critic_mem_lstm_hid_sizes 128 --critic_cur_feature_hid_sizes 128 --critic_post_comb_hid_sizes 128 --actor_mem_pre_lstm_hid_sizes 128 --actor_mem_lstm_hid_sizes 128 --actor_cur_feature_hid_sizes 128 --actor_post_comb_hid_sizes 128

    Continuing Algorithm Training (include --load_from_directory)

    $ python main.py "./the_previously_created_training_directory/" --load_from_directory

    Rendering An Algorithm From Created Training Directory (include --render)

    $ python main.py "./the_previously_created_training_directory/" --render

Author:
    Jordan Cramer

Date:
    2023-09-09
"""
import argparse
from ma_algorithm_runner import AlgorithmRunner


if __name__ == '__main__':
    """ Run a multi-agent trial from the command-line using argparse to parse command-line arguments. """

    """ Set up command-line argument parsing. """
    # parser setup
    description = 'Runs a multi-agent reinforcement learning (MARL) algorithm.'
    parser = argparse.ArgumentParser(description=description)

    # the directory to save everything to
    parser.add_argument('directory', type=str)

    # if --render is not included, then the value is False
    #
    # if --render is set to be True, then the algorithm will be loaded from the directory and rendered
    # all other arguments will be ignored (N/A)
    parser.add_argument('--render', action='store_true')

    # if --load_from_directory is not included, then the value is False
    parser.add_argument('--load_from_directory', action='store_true')
    
    # Create an argument parser to accept POMDP and POMDP type as arguments
    #scenarios are: 1- random_sensor_missing 2-landmark_missing_position 3-random_noise
    
    parser.add_argument('--pomdp', action='store_true', help='Enable POMDP')
    parser.add_argument('--pomdp_type', type=str, default='random_sensor_missing', help='Specify POMDP type')

    # environment, algorithm, training, and logging
    parser.add_argument('--algorithm_name', type=str, default='ma_ddpg')
    parser.add_argument('--env_name', type=str, default='simple_adversary_v3')
    parser.add_argument('--training_steps', type=int, default=1_000_000)
    parser.add_argument('--checkpoint_interval', type=int, default=5_000)
    parser.add_argument('--log_interval', type=int, default=10_000)
    parser.add_argument('--evaluation_interval', type=int, default=1_000)
    parser.add_argument('--evaluation_episodes', type=int, default=25)
    parser.add_argument('--start_steps', type=int, default=10_000)
    parser.add_argument('--update_after', type=int, default=5_000)
    parser.add_argument('--update_every', type=int, default=100)
    parser.add_argument('--optimization_steps', type=int, default=1)

    # TD3 relevant arguments
    parser.add_argument('--delay_interval', type=int, default=2)

    # network parameters
    parser.add_argument('--action_noise_std', type=float, default=0.1)
    parser.add_argument('--actor_learning_rate', type=float, default=1e-4)
    parser.add_argument('--critic_learning_rate', type=float, default=1e-3)
    parser.add_argument('--buffer_size', type=int, default=100_000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--discount_factor', type=float, default=0.95)
    parser.add_argument('--tau', type=float, default=0.01)

    # network parameters
    # DDPG TD3 relevant arguments
    parser.add_argument('--actor_layer_1_size', type=int, default=64)
    parser.add_argument('--actor_layer_2_size', type=int, default=64)
    parser.add_argument('--critic_layer_1_size', type=int, default=64)
    parser.add_argument('--critic_layer_2_size', type=int, default=64)

    # network parameters
    # LSTM-TD3 relevant arguments
    parser.add_argument('--max_history_length', type=int, default=5)
    # Note (specifying the layers and layer sizes in the architecture):
    # nargs="+" means 1 or more values can be specified (example: --critic_mem_lstm_hid_sizes 128 64 32)
    # nargs="?" means 0 or 1 values can be specified (example: --actor_cur_feature_hid_sizes)
    # nargs="*" means 0 or more values can be specified
    parser.add_argument('--critic_mem_pre_lstm_hid_sizes', type=int, nargs="+", default=[128])
    parser.add_argument('--critic_mem_lstm_hid_sizes', type=int, nargs="+", default=[128])
    parser.add_argument('--critic_mem_after_lstm_hid_size', type=int, nargs="+", default=[])
    parser.add_argument('--critic_cur_feature_hid_sizes', type=int, nargs="+", default=[128, 128])
    parser.add_argument('--critic_post_comb_hid_sizes', type=int, nargs="+", default=[128])
    parser.add_argument('--actor_mem_pre_lstm_hid_sizes', type=int, nargs="+", default=[128])
    parser.add_argument('--actor_mem_lstm_hid_sizes', type=int, nargs="+", default=[128])
    parser.add_argument('--actor_mem_after_lstm_hid_size', type=int, nargs="+", default=[])
    parser.add_argument('--actor_cur_feature_hid_sizes', type=int, nargs="+", default=[128, 128])
    parser.add_argument('--actor_post_comb_hid_sizes', type=int, nargs="+", default=[128])

    # LSTM-TD3 relevant arguments
    parser.add_argument('--target_noise', type=float, default=0.1)
    parser.add_argument('--target_noise_clip', type=float, default=0.5)

    # Don't save replay buffer when I am just running hyperparameter tests
    # It wastes hard-drive space and time with writing to disk
    parser.add_argument('--save_replay_buffer', action='store_true')

    # I thought that scaling the lstm gradients (so they learn faster),
    # might make the ma_lstm_td3 learn faster. I decided to try it out here.
    parser.add_argument('--scale_lstm_gradients', action='store_true')
    parser.add_argument('--scale_factor_lstm_gradients', type=float, default=2.0)


    # LSTM-TD3 relevant arguments
    #
    # For MA-LSTM-TD3, these arguments should generally be set True
    #
    # when the arguments are not included in the command-line argument:
    # they are set to True
    #
    # when the arguments are included in the command-line argument:
    # they are set to False
    # parser.add_argument('--use_double_critic', default=True, action='store_false')
    # parser.add_argument('--use_target_policy_smoothing', default=True, action='store_false')
    # parser.add_argument('--critic_hist_with_past_act', default=True, action='store_false')
    # parser.add_argument('--actor_hist_with_past_act', default=True, action='store_false')

    # LSTM-TD3 is going to be set to use double critics and target policy smoothing all the time
    # These are key improvements that are part of the TD3 algorithm and will lead to better performance
    # Similar to the LSTM-TD3 paper Ablation Study
    # These could be coded to be turned off later to see how performance is impacted if they are missing
    #
    # parser.add_argument('--use_double_critic', type=str2bool, nargs='?', const=True, default=True, help="Using double critic")
    # parser.add_argument('--use_target_policy_smooth', type=str2bool, nargs='?', const=True, default=True, help="Using target policy smoothing")

    # The Ablation Study also tried some different architectures,
    # where the actions were not concatenated with the observations before the LSTM unit.
    # This could also be tried out later if desired.
    #
    # parser.add_argument('--critic_hist_with_past_act', type=str2bool, nargs='?', const=True, default=True)
    # parser.add_argument('--actor_hist_with_past_act', type=str2bool, nargs='?', const=True, default=True)

    # parse the command-line arguments
    args = parser.parse_args()
    arguments: dict = vars(args)
    print(f'... running for environment {args.env_name} ...')

    # interpret without current feature extraction
    if args.critic_cur_feature_hid_sizes is None:
        args.critic_cur_feature_hid_sizes = []
    if args.actor_cur_feature_hid_sizes is None:
        args.actor_cur_feature_hid_sizes = []

    # the hidden layer sizes must be lists of integers and not just integers (example: [10] not 10)
    def convert_to_list(variable):
        """ Convert integers to lists. Lists stay lists. If not an integer or a list, raise an exception. """
        if isinstance(variable, int):
            return [variable]
        elif isinstance(variable, list):
            return variable
        else:
            raise TypeError(f"The parsed argument is not an integer or a list. Type: {type(variable)}")

    args.critic_mem_pre_lstm_hid_sizes = convert_to_list(args.critic_mem_pre_lstm_hid_sizes)
    args.critic_mem_lstm_hid_sizes = convert_to_list(args.critic_mem_lstm_hid_sizes)
    args.critic_mem_after_lstm_hid_size = convert_to_list(args.critic_mem_after_lstm_hid_size)
    args.critic_cur_feature_hid_sizes = convert_to_list(args.critic_cur_feature_hid_sizes)
    args.critic_post_comb_hid_sizes = convert_to_list(args.critic_post_comb_hid_sizes)
    args.actor_mem_pre_lstm_hid_sizes = convert_to_list(args.actor_mem_pre_lstm_hid_sizes)
    args.actor_mem_lstm_hid_sizes = convert_to_list(args.actor_mem_lstm_hid_sizes)
    args.actor_mem_after_lstm_hid_size = convert_to_list(args.actor_mem_after_lstm_hid_size)
    args.actor_cur_feature_hid_sizes = convert_to_list(args.actor_cur_feature_hid_sizes)
    args.actor_post_comb_hid_sizes = convert_to_list(args.actor_post_comb_hid_sizes)

    """
    Do one of 3 things based on the passed in arguments.
    
    1. Render an already trained algorithm. (--render)
    2. Load and continue training an algorithm. (--load_from_directory)
    3. Start training an algorithm from the beginning.
    """
    if args.render is True:
        print(f'... rendering ...')
        runner = AlgorithmRunner(arguments=arguments)
        runner.render_algorithm()
    elif args.load_from_directory is True:
        print('... loading from directory and continuing training ...')
        runner = AlgorithmRunner(arguments=arguments)
        runner.train_algorithm()
    else:
        print(f'... starting training for {args.algorithm_name} ...')
        runner = AlgorithmRunner(arguments=arguments)
        runner.train_algorithm()

    print(f'... running for environment {args.env_name} ...')
    
    # if args.pomdp is True:
    #     print(f'...runing for pomdp environment {args.env_name}')