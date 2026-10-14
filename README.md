# Multi-Agent LSTMTD3
Python Code for the MA-LSTMTD3 algorithm developed for the paper : " Memory Augmented Multi-Agent Reinforcement Learning for Cooperative Environment"
https://link.springer.com/chapter/10.1007/978-3-031-81596-6_8

## Description 
To run three deterministic algorithms for multi-agent reinforcement learning:  
1.Multi-Agent Deep Deterministic Policy Gradient (MADDPG)  
2.Multi-Agent Twin Delayed Deep Deterministic Policy Gradient (MATD3)  
3.Multi-Agent Long Short-Term Memory Twin Delayed Deep Deterministic Policy Gradient (MALSTMTD3)  

These algorithms are designed to work with both Markov Decision Processes (MDP) and Partially Observable Markov Decision Processes (POMDP) versions of the tasks. The specific environments used are the PettingZoo Multi-Particle Environments (MPE Environments).

## Running the Code

To train an algorithm from scratch (excluding rendering and loading from directory), run the following command from the command line:
python main.py runs/simple_spread_lstm --algorithm_name ma_lstm_td3 --env_name simple_spread_v3

## Code Contributors  
Based on an initial implementation by Jordan Cramer. Maryam Kia extended the code to the full experiments.

## References
A major code reference for the MADDPG Agents was as follows:
* License: MIT License
* Repo: https://github.com/philtabor/Multi-Agent-Reinforcement-Learning/tree/main

A major code reference for the LSTM-TD3 agent was as follows:
* License: MIT License
* Repo: https://github.com/LinghengMeng/LSTM-TD3

* License: MIT License
* Repo: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch

* Paper Title: Partial Observability during DRL for Robot Control
* Paper: https://arxiv.org/pdf/2209.04999.pdf
