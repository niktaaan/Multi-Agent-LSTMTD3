# Program

## Description
I wrote this program to practice using OpenAI Gymnasium MuJoCo environments.

I wanted to follow the paper and create an LSTM-TD3 reinforcement learning agent to test with Partially Observable Markov Decision Processes (POMDPs).

I followed the code outlined by the paper. The paper followed the original TD3 code from the OpenAI Spinning Up website.

Paper:

https://arxiv.org/abs/2102.12344

Paper Code:

https://github.com/LinghengMeng/LSTM-TD3/blob/main/lstm_td3/env_wrapper/pomdp_wrapper.py

Spinning Up - Code:

https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch

Spinning Up - Normal TD3 Algorithm:

https://spinningup.openai.com/en/latest/algorithms/td3.html

Parts of the code were complicated or unnecessary for my purposes. These parts were rewritten. In particular, I tried to increase the amount of encapsulation (moving plenty of relevant functions, data, and code into the agent class).

## File: runner_lstm_td3.py
This is the main file where I run the lstm-td3 algorithm with various settings and hyperparameters specified.

## File: programming_practice/practice_openai_gymnasium_observation_wrapper.py
I practiced creating POMDPs by modifying the environment observations/states using the ObservationWrapper class given by OpenAI Gymnasium. This is a simple, self-contained program with plenty of comments.

## File: programming_practice/practice_openai_gymnasium_mujoco.py
I practiced using the OpenAI Gymnasium MuJoCo environments. It is a very simple, self-contained program with comments explaining how to run a MuJoCo environment.

## File: environment/score_storage.py
This file contains a simple class for storing time step rewards and storing episode scores. This will be useful when I want to plot the performance of the agent in a simple way.

## File: environment/mdp_wrapper.py
This is a simple class for adding additional functionality to OpenAI Gymnasium Environments. These classes are called wrapper classes because they wrap around the exist code and add additional functionality. For this wrapper class, I am basically just recording the rewards and scores at each time step.

## File: environment/pomdp_wrapper.py
This is a more complicated wrapper class. The main purpose of this class is modifying the environment observations/states to create POMDPs. The states can be modified in many ways. The file contains plenty of comments and documentation going into more specifics about this. Also, try reading the paper to see results.

## File: algorithm/agent_lstm_td3.py
A more complicated td3 agent that makes use of the agent's history of environment observations and actions. This agent uses a more complicated version of experience replay that stores histories of observation-action pairs.

## Code Refactoring
I refactored code (for the LSTM-TD3 agent) from the paper and from OpenAI Spinning Up. In summary, what I changed and the reasons why are listed below.

(1) Code encapsulation is enhanced.
* Many functions, the optimization of weights, the replay buffer, and learning was all outside of the agent class.
* I decided to bring all of this functionality into the agent class.
* The data and functions relevant to the rl agent should have been encapsulated by the agent class.

(2) I did my best to refactor the code to make it more simple and organized. (Object-oriented Paradigm)
* There seemed to be significant amount of python inner function abuse.
* There seemed to be a mix of object-oriented and procedural programming paradigms.
* In general, I tried to make everything more object-oriented.

(3) I did not need argparse for command-line arguments.
* A command-line interface for running RL trials is a good thing, but it was not necessary when refactoring the code.
* It could be added again in the future after the code is refactored.
* I think it would be best to reintroduce argparse if many trials will be run in the future, but it was just difficult to refactor and test the code quickly while also maintaining argparse.

(4) Added more comments and documentation.
* I added a lot of comments to the code and function documentation.
