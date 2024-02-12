"""
Description:
    OpenAI purchased the Mujoco Physics Engine and made it free and open source.
    It is now supposed to be available and free to use with OpenAI Gymnasium.

    https://gymnasium.farama.org/environments/mujoco/

    Thus, I will try to get a Mujoco environment working here to continue doing reinforcement learning
    with these more sophisticated/complicated tasks/environments.

    I'll try a simple Mujoco environment like "Ant-v4".

    https://gymnasium.farama.org/environments/mujoco/ant/

    For some of the environments that are not working due to some MuJoCo version problem,
    I have found this solution online.

    https://stackoverflow.com/questions/76258775/openai-gym-mojoco-walker2d-v4-model-global-cordinates-error

Author:
    Jordan Cramer

Date:
    2023-07-22
"""
import gymnasium as gym


""" Try to run some simple sample code and see if my understanding is correct. """
if __name__ == '__main__':
    # making the environment works just like normal for OpenAI Gymnasium environments
    # including render_mode='human' will animate the environment with a graphics window
    # env = gym.make("Ant-v4", render_mode='human')

    # the environment can be run without animation of course
    env = gym.make("Ant-v4")

    # here are some other environments that can be tried after commenting out the "Ant-v4" line above
    # env = gym.make("HalfCheetah-v4", render_mode='human')  # seems to work
    # env = gym.make("Walker2d-v4", render_mode='human')  # doesn't work with the current MuJoCo version
    # env = gym.make("Hopper-v4", render_mode='human')  # doesn't work with the current MuJoCo version
    # env = gym.make("Humanoid-v4", render_mode='human')  # seems to work
    # env = gym.make("InvertedPendulum-v4", render_mode='human')  # seems to work
    env = gym.make("InvertedDoublePendulum-v4", render_mode='human')
    # env = gym.make("Reacher-v4", render_mode='human')  # seems to work
    # env = gym.make("Pusher-v4", render_mode='human')  # seems to work

    # the environment must be reset before running
    observation, info = env.reset()

    # the observation for 'Ant-v4' is of type <class 'numpy.ndarray'>
    # the shape is (27,) for 'Ant-v4'
    print('Observation Type:', type(observation))
    print('Observation Shape:', observation.shape)
    print('Action Space Type:', type(env.action_space))
    print('Action Space Shape:', env.action_space.shape)
    print('Action Space:', env.action_space)
    print('Action Space Low:', env.action_space.low[0])
    print('Action Space High:', env.action_space.high[0])
    print('Observation Space Type:', type(env.observation_space))
    print('Observation Space Shape:', env.observation_space.shape)
    print('Observation Space:', env.observation_space)
    print()

    # here we will run the environment for 1000 time steps
    number_of_time_steps = 0  # keep track of the number of time steps during the episodes
    for i in range(1000):
        # increment the number of time steps for the current episode
        number_of_time_steps += 1

        # here the action space is being randomly sampled since there is no policy we are using yet
        action = env.action_space.sample()

        # now we use the sample action to step forward by one time step in the environment to reach the next state
        observation, reward, terminated, truncated, info = env.step(action)

        # if the episode terminates or truncates (I believe ant truncates at 1000 time steps)
        # rest the environment
        if terminated or truncated:
            print('Time Steps:', number_of_time_steps)
            number_of_time_steps = 0
            if terminated:
                print('Episode Terminated')
            elif truncated:
                print('Episode Truncated')
            observation, info = env.reset()

    print()
    print('Closing the Environment')
    env.close()
    print('The Environment Was Closed')

    # note: there might be an error message with the rendering at the end of the program here
    # I looked online and I believe this might be a minor error caused by Python's OpenGL
    # maybe the rendering window context is not deleted at the end of the gymnasium implementation
    # just ignoring it is fine since it just happens when the script terminates and doesn't affect anything
