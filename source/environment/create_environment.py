
from pettingzoo.mpe import simple_adversary_v3
from pettingzoo.mpe import simple_spread_v3
from pettingzoo.mpe import simple_speaker_listener_v4


def create_environment(env_name: str, render: bool = False):
    """

    Args:
        env_name (str): The environment to be created"

        render (bool): If this bool is set to True, then the environment will be created for rendering the animation.

    Returns:
        The created environment.
    """
    # create the correct environment
    if env_name == 'simple_adversary_v3':
        env = simple_adversary_v3.parallel_env(
            N=2,  # number of cooperative agents and landmarks (default: 2)
            max_cycles=25,  # specify the maximum number of agent actions/time steps before episode truncation
            continuous_actions=True,  # environment discrete/continuous action space flag
            render_mode='human' if render else None
        )
        _, _ = env.reset()
    elif env_name == 'simple_spread_v3':
        env = simple_spread_v3.parallel_env(
            N=3,  # number of agents and landmarks
            local_ratio=0.5,  # reward weight for local agent rewards
            max_cycles=25,
            continuous_actions=True,
            render_mode='human' if render else None
        )
        _, _ = env.reset()
    elif env_name == 'simple_speaker_listener_v4':
        env = simple_speaker_listener_v4.parallel_env(
            max_cycles=25,
            continuous_actions=True,
            render_mode='human' if render else None
        )
        _, _ = env.reset()
    # raise an error if the environment hasn't been handled above or mistyped
    else:
        raise ValueError(f'The environment name "{env_name}" is not currently handled by the create_environment function. '
                         f'Was the environment name mistyped?')

    return env
