"""
Description:
    A very simple class for storing time step rewards and episode scores.

Author:
    Jordan Cramer

Date:
    2023-07-24
"""


class ScoreStorage:
    """ A very simple class for storing time step rewards and episode scores. """

    def __init__(self):
        self.scores = []  # list that stores episode scores
        self.total_reward = 0  # stores the current total reward received by the agent during the current episode
        self.last_reward = 0  # stores the last reward received by the agent

    def add_score(self):
        """ Add a score to the list of scores. """
        self.scores.append(self.total_reward)

    def add_reward(self, reward: float):
        """ Add the reward of the current time step to the total return of the episode. """
        self.last_reward = reward
        self.total_reward += self.last_reward

    def episode_reset(self):
        """ Record the episode score and reset for the next episode. """
        self.add_score()
        self.total_reward = 0
        self.last_reward = 0
