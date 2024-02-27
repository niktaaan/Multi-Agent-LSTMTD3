# Multi-Agent LSTMTD3
Python Code for MA-LSTMTD3 algorithm 

## Description 
To run three deterministic algorithms for multi-agent reinforcement learning:  
1.Multi-Agent Deep Deterministic Policy Gradient (MADDPG)  
2.Multi-Agent Twin Delayed Deep Deterministic Policy Gradient (MATD3)  
3.Multi-Agent Long Short-Term Memory Twin Delayed Deep Deterministic Policy Gradient (MALSTMTD3)  

For both Markov Decision Processes(MDP) and Partially Observable Markov Decision Processes (POMDP) versions of the tasks.
The specific environments are the PettingZoo Multi-Particle Environments (MPE Environments):  
1.simple_adversary_v3  
2.simple_spread_v3  
3.simple_speaker_listener_v4  

  main.py  
  Script to train multi agent algorithm or render environment.
  Arguments are parsed from the command line.

  ma_algorithm_runner.py  
  This module containing the AlgorithmRunner class, a high level class that organizes the algorithms, environments, and data logging.  

  create_algorithm.py
  Creates the appropriate multi-agent algorithm based on the provided arguments and hyperparameters.

  ma_ddpg.py  
  Wrapper class that coordinates agents in the multi-agent DDPG Algorithm.

  agent_ddpg.py  
  The Deep Deterministic Policy Gradient (DDPG) agent required for MADDPG algorithm.


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
