import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def combined_rewards_plot(rewards_list_1, rewards_list_2, labels, environment_type, window_size=10):
    """
    Plots rewards per episode for two separate reward lists with optional smoothing.

    Args:
        rewards_list_1 (dict): A dictionary where keys are episodes and values are rewards for the first set.
        rewards_list_2 (dict): A dictionary where keys are episodes and values are rewards for the second set.
        labels (tuple): A tuple containing the labels for the two reward lists (e.g., ('Model A', 'Model B')).
        environment_type (str): The type of environment (for plot title).
        window_size (int, optional): The window size for moving average smoothing. Default is 10.
    """
    # Extract episode and reward values for both reward lists
    episodes_1 = np.array(list(rewards_list_1.keys())).flatten()
    rewards_1 = np.array(list(rewards_list_1.values())).flatten()
    
    episodes_2 = np.array(list(rewards_list_2.keys())).flatten()
    rewards_2 = np.array(list(rewards_list_2.values())).flatten()
    
    plt.figure(figsize=(12, 7))

    # Plot first reward list
    plt.plot(episodes_1, rewards_1, marker='o', linestyle='-', markersize=4, label=f'{labels[0]} Rewards')

    # Plot moving average for the first reward list
    if window_size > 1:
        moving_avg_1 = np.convolve(rewards_1, np.ones(window_size) / window_size, mode='valid')
        plt.plot(episodes_1[:len(moving_avg_1)], moving_avg_1, linestyle='--', color='r', alpha=0.8, label=f'{labels[0]} Moving Avg')

    # Plot second reward list
    plt.plot(episodes_2, rewards_2, marker='s', linestyle='-', markersize=4, label=f'{labels[1]} Rewards')

    # Plot moving average for the second reward list
    if window_size > 1:
        moving_avg_2 = np.convolve(rewards_2, np.ones(window_size) / window_size, mode='valid')
        plt.plot(episodes_2[:len(moving_avg_2)], moving_avg_2, linestyle='--', color='b', alpha=0.8, label=f'{labels[1]} Moving Avg')

    plt.title(f'Comparison of Rewards per Episode: {environment_type}', fontsize=20)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    return plt

def load_variables(file_path):
    try:
        # Load the variables from the specified file
        loaded_variables = torch.load(file_path)
        print(f"Loaded variables from {file_path}.")
        return loaded_variables

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}.")
        return None

    except Exception as e:
        print(f"An error occurred while loading variables: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    dir_path = 'saved_variables'

    # Load DDQN variables
    ddqn_rewards = torch.load(os.path.join(dir_path, 'ddqn_rewards.pkl'))
    ddqn_losses = torch.load(os.path.join(dir_path, 'ddqn_losses.pkl'))
    ddqn_rewards_per_episode = torch.load(os.path.join(dir_path, 'ddqn_rewards_per_episode.pkl'))
    ddqn_epsilon_values = torch.load(os.path.join(dir_path, 'ddqn_epsilon_values.pkl'))

    # Load DQN variables
    dqn_rewards = torch.load(os.path.join(dir_path, 'dqn_rewards.pkl'))
    dqn_losses = torch.load(os.path.join(dir_path, 'dqn_losses.pkl'))
    dqn_rewards_per_episode = torch.load(os.path.join(dir_path, 'dqn_rewards_per_episode.pkl'))
    dqn_epsilon_values = torch.load(os.path.join(dir_path, 'dqn_epsilon_values.pkl'))
    labels = ('DQN', 'DDQN')
    
    environment_type = 'Traffic Simulation'
    window_size = 10

    # Similarly load for A2C and A3C
    a3c_rewards_per_episode = torch.load(os.path.join(dir_path, 'a3c_rewards.pkl'))
    a2c_rewards_per_episode = torch.load(os.path.join(dir_path, 'a2c_rewards.pkl'))

    labels2 = ('A2C', 'A3C')
    # a3c_average_rewards = torch.load(os.path.join(dir_path, 'a3c_average_rewards.pkl'))


    combined_rewards_plot(dqn_rewards_per_episode, ddqn_rewards_per_episode, labels, environment_type, window_size)
    combined_rewards_plot(a3c_rewards_per_episode, a2c_rewards_per_episode, labels2, environment_type, window_size)