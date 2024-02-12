# Program

## Description: Agents and Algorithms
This program is for trying out the MADDPG algorithm with different reinforcement learning agents.

I hope to try out the following agents and algorithms,

1. DDPG
2. TD3
3. LSTM-TD3
4. MADDPG
5. MA-TD3
6. MA-LSTM-TD3 (Main Goal)
7. Transformer-TD3 (Maybe)
8. MA-Transformer-TD3 (Maybe)

## Environments
I hope to try the agents and algorithms for Markov Decision Processes (MDPs) and Partially Observable Markov Decision Processes (POMDPs).

The strength of LSTM-TD3 is that by adding the LSTM (and history sensitivity), the agents are better at dealing with POMDPs. It would be a good part of a possible paper to observe how the Multi-agent algorithm deals with POMDPs.

The specific environments that could be tested would be the PettingZoo Multi-Particle Environments (MPE Environments).

So far, the following 3 environments seem to run well.

1. simple_adversary_v3
2. simple_spread_v3
3. simple_speaker_listener_v4

## main.py
Mostly everything is programmed to run from the main.py file. You will find all the relevant commands and examples in that file for running training and rendering from the command-line.

## ma_algorithm_runner.py
This is the module containing the AlgorithmRunner class that basically runs all training and rendering of the algorithms. It is a high level class that organizes the algorithms, environments, and data logging.

## render_ensemble.py
I had the idea that I would like to try training several different algorithms and then using them as ensemble. The trained algorithms are loaded from their directories and evaluated.

Actions taken in the environment are the averages of all the algorithms. I did not see any performance benefit from doing this simplistic version of an ensemble for the TD3 algorithm. I thought that maybe the score or standard deviation of scores would be better. Neither were better than the individually trained multi-agent algorithms.

However, I wonder if the ensemble is less brittle in the following sense. Maybe the individually trained algorithms can catastrophically failure for some episodes while the ensemble would be less likely to do so. However, I would have hoped to have seen this in a better standard deviation for scores, which I did not see for the ensemble.

## Directory: example_code
I placed some code here that I thought was useful when I was learning how to use PettingZoo environments and concatenate batches of multi-agent critic observation-action states.

### example_code/environment_simple_adversary_v3.py
I wrote all the code below to test my understanding of how multi-agent PettingZoo environments work.
It is now a good code example of how they work.

However, I learned something important. There are two different kinds of PettingZoo environments.

(1) The first environment type is Agent Environment Cycle (AEC)

https://pettingzoo.farama.org/api/aec/

This type of environment cycles through each agent in sequence.
Agent 1 chooses and takes an action.
Agent 2 chooses and takes an action.
Agent 3 ...
Each agent must wait for the previous agent to choose and finish their action first.

(2) The second environment type is Parallel Environments

https://pettingzoo.farama.org/api/parallel/

All agents choose actions and perform those actions in parallel rather than in a sequence.

### example_code/observation_action_batch_concatenation_test.py
It was rather confusing for me to concatenate batches of observation and actions for the critic states for the multi-agent algorithms. I managed to figure it out by creating this example script for myself.

## Directory: old_code
This is just a directory where I dumped old code as I was refactoring. It was just in case I needed to reference these files after making some kind of mistake. These files are mostly obsolete and can just be deleted.

## Directory: source
All the source code is located in this directory

### Directory: source/environment/
For the current program, I made a simple function to create environments with simple presets. In the future, the code for the environment will likely need to be improved upon so that POMDPs can also be tested with the multi-agent LSTM-TD3 algorithm.

To see how POMDPs can be created with environment wrappers, my "LSTM_TD3_Gymnasium_MuJoCo" project is a good example for single agent environments.

### Directory: source/multi_agent_algorithm/
All the code for the MADDPG and MATD3 algorithms is located here. It is fairly well-documented and consistent code.

### Directory: source/single_agent_lstm_td3/
Before attempting to write the multi-agent lstm algorithm I decided it was necessary to fully refactor the single agent LSTM-TD3 algorithm, so that it was consistent with the code for the DDPG and TD3 algorithms. Otherwise, it might be hard to integrate due to all the differences.

This is a mostly self-contained code example of the single agent LSTM-TD3 algorithm running for an example OpenAI Gymnasium MuJoCo environment.

### Directory: source/multi_agent_lstm_td3/
Since many things are so different for the MALSTMTD3 algorithm compared with the MADDPG and MATD3 algorithms (replay buffer), I decided to try to first create the multi-agent algorithm separately. The algorithm can later be integrated into the other code once it is working in isolation here.

I thought it would be easier to do this than try to immediately code the complicated algorithm and figure out how to make the existing multi-agent replay buffer work for the new MALSTMTD3 code at the same time.

### Directory: source/utility/
I wrote helpful code for data logging here.

#### File: source/utility/data_manager.py
This is a class for exporting plots and .csv files to directories.

#### File: source/utility/dynamic_array.py
I wrote a simple, efficient dynamic array class.

Python does not have an efficient dynamic array built-in. A dynamic array is simply an array that can grow in size when it needs to store more elements.

Python lists can grow in size with the function my_list.append(new_element), but this is actually very inefficient since every single time all the data is copied over to the new list with size+1. This is very inefficient when the length of the list is large.

Thus, I created a dynamic_array class (using numpy under the hood) that will double in size when the storage capacity needs to increase. This makes it much more efficient than appending to Python lists and Numpy arrays. The number of complete copies of all data is reduced.

## LSTM-TD3 Paper Ablation Study

In the Ablation Study section of the paper they tested different architectures and algorithms in the following ways. To figure out the importance of different parts of the algorithm and architecture, they would remove these parts/features and evaluate the performance of the algorithm.

Some of the things they removed and tested are the following.
1. TD3 double critic usage
2. TD3 target policy smoothing
3. Concatenating observation histories with actions (adding past actions) before the LSTM unit
4. Having a fully 

For the initial MA-LSTM-TD3 implementation, I will try to use the architecture and configuration that gave the best possible performance.

## Future Work
It might be interesting to take this as an opportunity to try out (and learn about) other popular MARL algorithms to compare them with MADDPG and its variants.

Example MARL algorithms
* QMIX
* COMA

It would be good to try different architectures like the ablation study in the LSTM-TD3 paper.

### Future Work: List of Things to Look Into (Rough Draft)
* multi-step TD(n) or eligibility traces algorithms can be implemented and compared with the MA(DDPG/TD3/LSTM-TD3) algorithms.
* batch normalization
* values for TD3 smoothing clipping (normally the clipping is done, but it doesn't have to be done, or the clipping bounds could be different)
* hyperparameter tuning algorithm (I would try out something like several rounds of random search first)
* tests can take a long time, so running multiple tests (useful when hyperparameter tuning) on compute Canada would be a good exercise
* create some POMDP environments and test out the multi-agent lstm-td3 algorithm
* try out other popular multi-agent algorithms like QMIX
* maybe prioritized and hindsight experience replay
* maybe try out MAPPO
* try making the number of hidden layers in the MADDPG and MATD3 actor and critic networks a tunable hyperparameter using a tuple argument [100,100] might create 2 hidden layers with 100 neurons for example

## Automatic Documentation Generation (Sphinx and Pdocs)
I tried to use Sphinx and ReStructuredText for automatic documentation generation. Sphinx encountered errors due to the PettingZoo code for some reason, and I was unable to resolve the problem.

I switched to Pdoc and Google Style Docstrings and Markdown, since it was a simple alternative.

https://pdoc.dev/

To generate documentation, open up the command line in the directory with the script and use the following command.

$ pdoc --html my_script.py

It will generate a folder called HTML in the same directory with the documentation file in HTML format. There are other formats and options as well.

I'm not sure how Pdoc works with a folder hierarchy. It might require the folders to be Python modules with "\_\_init\_\_.py" files.

## References
A major code reference for the MADDPG reference (which was used when writing my code) was the following.

* License: MIT License
* Repo: https://github.com/philtabor/Multi-Agent-Reinforcement-Learning/tree/main

A major code reference for the LSTM-TD3 agent was the following.

* License: MIT License
* Repo: https://github.com/LinghengMeng/LSTM-TD3

The code that LinghengMeng originally used was from OpenAI SpinningUp.

* License: MIT License
* Repo: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch

Another paper by Lingheng Meng in 2022 concerning more POMDP and DRL for Robot Control.

* Paper Title: Partial Observability during DRL for Robot Control
* Paper: https://arxiv.org/pdf/2209.04999.pdf
