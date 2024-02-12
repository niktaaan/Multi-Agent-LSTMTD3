# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 13:00:17 2023

@author: m.kiakojouri
"""


import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def plot_experiments(environment, directory_list, file_name, label_list, pomdp=False):
    # Create file paths
    csv_file_paths = [os.path.join(directory, 'data', file_name) for directory in directory_list]

    # Read data from CSV files
    df_list = [pd.read_csv(file_path) for file_path in csv_file_paths]
    # Determine the y-axis limits based on a common range for all plots
    common_min = min(df["Score Sum"].min() for df in df_list)
    common_max = max(df["Score Sum"].max() for df in df_list)
    
    # Determine the step size for the y-axis ticks
    step_size = 10  # Adjust as needed
    
    # Calculate the y-ticks based on the common range and step size
    y_ticks = range(int(common_min), int(common_max) + 1, step_size)

    # Create the plot
    plt.figure(figsize=(8, 5))
    for i, df in enumerate(df_list):
        label = label_list[i]
        plt.plot(df["Time Step"], df["Score Sum"],marker='o', linestyle='-', label=label)

    # Customize the plot
    if pomdp:
        plt.title(f"POMDP {environment}")
    else:
        plt.title(f"MDP {environment}")
    
    plt.xlabel("Step")
    plt.ylabel("Score")
    
    # Specify steps on the y-axis
    # Set a fixed range and steps for the y-axis
    plt.yticks(y_ticks)

    
    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

MDP_directory_list=[
    ["./tests/ma_ddpg/", "./tests/ma_td3/", "./tests/ma_lstm_td3/"],
    ["./tests/simple_speaker_listener_v4_ma_ddpg/", "./tests/simple_speaker_listener_v4_ma_td3/", "./tests/simple_speaker_listener_v4_ma_lstm_td3/"],
    ["./tests/Simple_adversary_v3_ma_ddpg/", "./tests/Simple_adversary_v3_ma_td3/", "./tests/Simple_adversary_v3_ma_lstm_td3/"],
]

Random_inint_list=[
    ["./tests/simple_speaker_listener_v4_ma_ddpg/", "./tests/simple_speaker_listener_v4_ma_td3/", "./tests/simple_speaker_listener_v4_ma_lstm_td3/", "./tests/R2simple_speaker_listener_v4_ma_lstm_td3/","./tests/R3simple_speaker_listener_v4_ma_lstm_td3/"]
   ]
#MDP    

POMDP_directory_list=[   
    ["./tests/POMDP_simple_speaker_listener_v4_ma_ddpg/", "./tests/POMDP_simple_speaker_listener_v4_ma_td3/", "./tests/POMDP_simple_speaker_listener_v4_ma_lstm_td3/"],
    ["./tests/POMDP_Simple_adversary_v3_ma_ddpg/", "./tests/POMDP_Simple_adversary_v3_ma_td3/", "./tests/POMDP_Simple_adversary_v3_ma_lstm_td3/"],
 ["./tests/POMDP_simple_spread_v3_ma_ddpg/", "./tests/POMDP_simple_spread_v3_ma_td3/", "./tests/POMDP_simple_spread_v3_ma_lstm_td3/"],
 ["./tests/RN_POMDP_simple_speaker_listener_v4_ma_ddpg/","./tests/RN_POMDP_simple_speaker_listener_v4_ma_td3/","./tests/RN_POMDP_simple_speaker_listener_v4_ma_lstm_td3/"],
 ["./tests/RN_POMDP_simple_adversary_v3_ma_ddpg/","./tests/RN_POMDP_simple_adversary_v3_ma_td3/","./tests/RN_POMDP_simple_adversary_v3_ma_lstm_td3/"],
 ["./tests/RN_POMDP_simple_spread_v3_ma_ddpg/", "./tests/RN_POMDP_simple_spread_v3_ma_td3/", "./tests/RN_POMDP_simple_spread_v3_ma_lstm_td3/"],
["./tests/HigherRN_POMDP_simple_adversary_v3_ma_ddpg/", "./tests/HigherRN_POMDP_simple_adversary_v3_ma_td3/", "./tests/HigherRN_POMDP_simple_adversary_v3_ma_lstm_td3/"],
["./tests/HigherRN_POMDP_simple_speaker_listener_v4_ma_ddpg/", "./tests/HigherRN_POMDP_simple_speaker_listener_v4_ma_td3/", "./tests/HigherRN_POMDP_simple_speaker_listener_v4_ma_lstm_td3/"]
]
 

label_list=[
    ["MA_DDPG", "MA_TD3", "MA_LSTM_TD3"],
    ["MA_DDPG", "MA_TD3", "MA_LSTM_TD3"],
    ["MA_DDPG", "MA_TD3", "MA_LSTM_TD3","R2MA_LSTM_TD3","R3MA_LSTM_TD3"],
]
#MDP    ]
plot_experiments("Simple_Spread", MDP_directory_list[0], "data_time_step_500000.csv", label_list[0])
plot_experiments("Simple_Speaker_Listener", MDP_directory_list[1], "data_time_step_500000.csv", label_list[1])  
plot_experiments("Simple_Adversary",MDP_directory_list[2], "data_time_step_500000.csv", label_list[0])  
 
#POMDP
plot_experiments("Simple_Spread", POMDP_directory_list[2], "data_time_step_500000.csv", label_list[0],pomdp=True)
plot_experiments("Simple_Speaker_Listener", POMDP_directory_list[0], "data_time_step_500000.csv", label_list[1],pomdp=True)  
plot_experiments("Simple_Adversary",POMDP_directory_list[1], "data_time_step_500000.csv", label_list[0],pomdp=True)   

plot_experiments("Simple_Spread", POMDP_directory_list[5], "data_time_step_300000.csv", label_list[0],pomdp=True)
plot_experiments("Simple_Speaker_Listener", POMDP_directory_list[3], "data_time_step_300000.csv", label_list[1],pomdp=True)  
plot_experiments("Simple_Adversary",POMDP_directory_list[4], "data_time_step_300000.csv", label_list[0],pomdp=True)   
  

plot_experiments("Simple_Adversary", POMDP_directory_list[6], "data_time_step_300000.csv", label_list[2],pomdp=True) 
plot_experiments("Simple_Speaker_Listener", POMDP_directory_list[7], "data_time_step_300000.csv", label_list[1], pomdp=True)  
 

# =============================================================================
# simple_spread
# =============================================================================
ddpg_directory="./tests/ma_ddpg/"
td3_directory="./tests/ma_td3/"
lstm_td3_directory="./tests/ma_lstm_td3/"

ddpg_csv_directory=os.path.join(ddpg_directory,'data')
ddpg_csv_file_name = "data_time_step_500000.csv"
ddpg_csv_file_path = os.path.join(ddpg_csv_directory,ddpg_csv_file_name)
td3_csv_directory=os.path.join(td3_directory,'data')
td3_csv_file_name = "data_time_step_500000.csv"
td3_csv_file_path = os.path.join(td3_csv_directory,td3_csv_file_name)
lstm_td3_csv_directory=os.path.join(lstm_td3_directory,'data')
lstm_td3_csv_file_name = "data_time_step_500000.csv"
lstm_td3_csv_file_path = os.path.join(lstm_td3_csv_directory, lstm_td3_csv_file_name)

ddpg_df = pd.read_csv(ddpg_csv_file_path)
td3_df = pd.read_csv(td3_csv_file_path)
lstm_td3_df = pd.read_csv(lstm_td3_csv_file_path)

plt.figure(figsize=(8, 5))  # Set the figure size
plt.plot(ddpg_df["Time Step"], ddpg_df["Score Sum"], marker='o', linestyle='-', label="MA_DDPG")
plt.plot(td3_df["Time Step"], td3_df["Score Sum"], marker='o', linestyle='-', label="MA_TD3")
plt.plot(lstm_td3_df["Time Step"], lstm_td3_df["Score Sum"], marker='o', linestyle='-', label="MA_LSTM-TD3")

# Customize the plot
plt.title("MDP Simple_Spread")
plt.xlabel("Step")
plt.ylabel("Score")

# Add legend
plt.legend()

# Show the plot
plt.show()

# =============================================================================
# simple_speaker_listener
# =============================================================================
ddpg_directory="./tests/simple_speaker_listener_v4_ma_ddpg/"
td3_directory="./tests/simple_speaker_listener_v4_ma_td3/"
lstm_td3_directory="./tests/simple_speaker_listener_v4_ma_lstm_td3/"

ddpg_csv_directory=os.path.join(ddpg_directory,'data')
ddpg_csv_file_name = "data_time_step_500000.csv"
ddpg_csv_file_path = os.path.join(ddpg_csv_directory,ddpg_csv_file_name)
td3_csv_directory=os.path.join(td3_directory,'data')
td3_csv_file_name = "data_time_step_500000.csv"
td3_csv_file_path = os.path.join(td3_csv_directory,td3_csv_file_name)
lstm_td3_csv_directory=os.path.join(lstm_td3_directory,'data')
lstm_td3_csv_file_name = "data_time_step_500000.csv"
lstm_td3_csv_file_path = os.path.join(lstm_td3_csv_directory, lstm_td3_csv_file_name)

ddpg_df = pd.read_csv(ddpg_csv_file_path)
td3_df = pd.read_csv(td3_csv_file_path)
lstm_td3_df = pd.read_csv(lstm_td3_csv_file_path)

plt.figure(figsize=(8, 5))  # Set the figure size
plt.plot(ddpg_df["Time Step"], ddpg_df["Score Sum"], marker='o', linestyle='-', label="MA_DDPG")
plt.plot(td3_df["Time Step"], td3_df["Score Sum"], marker='o', linestyle='-', label="MA_TD3")
plt.plot(lstm_td3_df["Time Step"], lstm_td3_df["Score Sum"], marker='o', linestyle='-', label="MA_LSTM-TD3")

# Customize the plot
plt.title("MDP Simple_Speaker_Listener")
plt.xlabel("Step")
plt.ylabel("Score")

# Add legend
plt.legend()

# Show the plot
plt.show()
# =============================================================================
# Simple Adversary
# =============================================================================
ddpg_directory="./tests/Simple_adversary_v3_ma_ddpg/"
td3_directory="./tests/Simple_adversary_v3_ma_td3/"
lstm_td3_directory="./tests/Simple_adversary_v3_ma_lstm_td3/"

ddpg_csv_directory=os.path.join(ddpg_directory,'data')
ddpg_csv_file_name = "data_time_step_500000.csv"
ddpg_csv_file_path = os.path.join(ddpg_csv_directory,ddpg_csv_file_name)
td3_csv_directory=os.path.join(td3_directory,'data')
td3_csv_file_name = "data_time_step_500000.csv"
td3_csv_file_path = os.path.join(td3_csv_directory,td3_csv_file_name)
lstm_td3_csv_directory=os.path.join(lstm_td3_directory,'data')
lstm_td3_csv_file_name = "data_time_step_500000.csv"
lstm_td3_csv_file_path = os.path.join(lstm_td3_csv_directory, lstm_td3_csv_file_name)

ddpg_df = pd.read_csv(ddpg_csv_file_path)
td3_df = pd.read_csv(td3_csv_file_path)
lstm_td3_df = pd.read_csv(lstm_td3_csv_file_path)

plt.figure(figsize=(8, 5))  # Set the figure size
plt.plot(ddpg_df["Time Step"], ddpg_df["Score Sum"], marker='o', linestyle='-', label="MA_DDPG")
plt.plot(td3_df["Time Step"], td3_df["Score Sum"], marker='o', linestyle='-', label="MA_TD3")
plt.plot(lstm_td3_df["Time Step"], lstm_td3_df["Score Sum"], marker='o', linestyle='-', label="MA_LSTM-TD3")

# Customize the plot
plt.title("MDP Simple_Adversary")
plt.xlabel("Step")
plt.ylabel("Score")

# Add legend
plt.legend()

# Show the plot
plt.show()
# =============================================================================
# Random Init
# =============================================================================

ddpg_directory="./tests/simple_speaker_listener_v4_ma_ddpg/"
td3_directory="./tests/simple_speaker_listener_v4_ma_td3/"
lstm_td3_directory="./tests/simple_speaker_listener_v4_ma_lstm_td3/"
R2lstm_td3_directory="./tests/R2simple_speaker_listener_v4_ma_lstm_td3/"
R3lstm_td3_directory="./tests/R3simple_speaker_listener_v4_ma_lstm_td3/"

ddpg_csv_directory=os.path.join(ddpg_directory,'data')
ddpg_csv_file_name = "data_time_step_200000.csv"
ddpg_csv_file_path = os.path.join(ddpg_csv_directory,ddpg_csv_file_name)
td3_csv_directory=os.path.join(td3_directory,'data')
td3_csv_file_name = "data_time_step_200000.csv"
td3_csv_file_path = os.path.join(td3_csv_directory,td3_csv_file_name)
lstm_td3_csv_directory=os.path.join(lstm_td3_directory,'data')
lstm_td3_csv_file_name = "data_time_step_200000.csv"
lstm_td3_csv_file_path = os.path.join(lstm_td3_csv_directory, lstm_td3_csv_file_name)

R2lstm_td3_csv_directory=os.path.join(R2lstm_td3_directory,'data')
R2lstm_td3_csv_file_name = "data_time_step_200000.csv"
R2lstm_td3_csv_file_path = os.path.join(R2lstm_td3_csv_directory, lstm_td3_csv_file_name)

R3lstm_td3_csv_directory=os.path.join(R3lstm_td3_directory,'data')
R3lstm_td3_csv_file_name = "data_time_step_200000.csv"
R3lstm_td3_csv_file_path = os.path.join(R3lstm_td3_csv_directory, lstm_td3_csv_file_name)

ddpg_df = pd.read_csv(ddpg_csv_file_path)
td3_df = pd.read_csv(td3_csv_file_path)
lstm_td3_df = pd.read_csv(lstm_td3_csv_file_path)
R2lstm_td3_df = pd.read_csv(R2lstm_td3_csv_file_path)
R3lstm_td3_df = pd.read_csv(R3lstm_td3_csv_file_path)

plt.figure(figsize=(8, 5))  # Set the figure size
plt.plot(ddpg_df["Time Step"], ddpg_df["Score Sum"], marker='o', linestyle='-', label="MA_DDPG")
plt.plot(td3_df["Time Step"], td3_df["Score Sum"], marker='o', linestyle='-', label="MA_TD3")
plt.plot(lstm_td3_df["Time Step"], lstm_td3_df["Score Sum"], marker='o', linestyle='-', label="MA_LSTM-TD3")
plt.plot(R2lstm_td3_df["Time Step"], R2lstm_td3_df["Score Sum"], marker='o', linestyle='-', label="R2MA_LSTM-TD3")
plt.plot(R3lstm_td3_df["Time Step"], R3lstm_td3_df["Score Sum"], marker='o', linestyle='-', label="R3MA_LSTM-TD3")


# Customize the plot
plt.title("MDP Simple_Speaker_Listener")
plt.xlabel("Step")
plt.ylabel("Score")

# Add legend
plt.legend()

# Show the plot
plt.show()
# =============================================================================
# POMDP
# =============================================================================

ddpg_directory="./tests/POMDP_simple_speaker_listener_v4_ma_ddpg/"
td3_directory="./tests/POMDP_simple_speaker_listener_v4_ma_td3/"
lstm_td3_directory="./tests/POMDP_simple_speaker_listener_v4_ma_lstm_td3/"


ddpg_csv_directory=os.path.join(ddpg_directory,'data')
ddpg_csv_file_name = "data_time_step_500000.csv"
ddpg_csv_file_path = os.path.join(ddpg_csv_directory,ddpg_csv_file_name)
td3_csv_directory=os.path.join(td3_directory,'data')
td3_csv_file_name = "data_time_step_500000.csv"
td3_csv_file_path = os.path.join(td3_csv_directory,td3_csv_file_name)
lstm_td3_csv_directory=os.path.join(lstm_td3_directory,'data')
lstm_td3_csv_file_name = "data_time_step_500000.csv"
lstm_td3_csv_file_path = os.path.join(lstm_td3_csv_directory, lstm_td3_csv_file_name)

ddpg_df = pd.read_csv(ddpg_csv_file_path)
td3_df = pd.read_csv(td3_csv_file_path)
lstm_td3_df = pd.read_csv(lstm_td3_csv_file_path)

plt.figure(figsize=(8, 5))  # Set the figure size
plt.plot(ddpg_df["Time Step"], ddpg_df["Score Sum"], marker='o', linestyle='-', label="MA_DDPG")
plt.plot(td3_df["Time Step"], td3_df["Score Sum"], marker='o', linestyle='-', label="MA_TD3")
plt.plot(lstm_td3_df["Time Step"], lstm_td3_df["Score Sum"], marker='o', linestyle='-', label="MA_LSTM-TD3")

# Customize the plot
plt.title("POMDP Simple_Speaker_Listener")
plt.xlabel("Step")
plt.ylabel("Score")

# Add legend
plt.legend()

# Show the plot
plt.show()

ddpg_directory="./tests/POMDP_Simple_adversary_v3_ma_ddpg/"
td3_directory="./tests/POMDP_Simple_adversary_v3_ma_td3/"
lstm_td3_directory="./tests/POMDP_Simple_adversary_v3_ma_lstm_td3/"

ddpg_csv_directory=os.path.join(ddpg_directory,'data')
ddpg_csv_file_name = "data_time_step_500000.csv"
ddpg_csv_file_path = os.path.join(ddpg_csv_directory,ddpg_csv_file_name)
td3_csv_directory=os.path.join(td3_directory,'data')
td3_csv_file_name = "data_time_step_500000.csv"
td3_csv_file_path = os.path.join(td3_csv_directory,td3_csv_file_name)
lstm_td3_csv_directory=os.path.join(lstm_td3_directory,'data')
lstm_td3_csv_file_name = "data_time_step_500000.csv"
lstm_td3_csv_file_path = os.path.join(lstm_td3_csv_directory, lstm_td3_csv_file_name)

ddpg_df = pd.read_csv(ddpg_csv_file_path)
td3_df = pd.read_csv(td3_csv_file_path)
lstm_td3_df = pd.read_csv(lstm_td3_csv_file_path)

plt.figure(figsize=(8, 5))  # Set the figure size
plt.plot(ddpg_df["Time Step"], ddpg_df["Score Sum"], marker='o', linestyle='-', label="MA_DDPG")
plt.plot(td3_df["Time Step"], td3_df["Score Sum"], marker='o', linestyle='-', label="MA_TD3")
plt.plot(lstm_td3_df["Time Step"], lstm_td3_df["Score Sum"], marker='o', linestyle='-', label="MA_LSTM-TD3")

# Customize the plot
plt.title("POMDP Simple_Adversary")
plt.xlabel("Step")
plt.ylabel("Score")

# Add legend
plt.legend()

# Show the plot
plt.show()