import torch.multiprocessing as mp
from torch.distributions import Categorical
import torch as T
import torch.nn.functional as F
from torch import nn
import gymnasium as gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import wandb
from tqdm import tqdm
import traci
import shutil
import sumo_rl

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")

class WanDB():
    def __init__(self, config, project, env, use_wandb):
        if use_wandb:
            wandb.init(project=project, config=config, name=env)
            self.config = wandb.config
            self.use_wandb = True
        else:
            self.config = config
            self.use_wandb = False

    def wandb_log(self, args):
        if self.use_wandb:
            wandb.log(args)
        else:
            print(args)

    def wandb_finish(self):
        if self.use_wandb:
            wandb.finish()



class Environment:
    def __init__(self, env_name, route_file, net_file, out_csv_name, render_mode='human', num_seconds=100000):
        self.env_name = env_name
        self.use_gui = True if render_mode == 'human' else False
        self.route_file = route_file
        self.net_file = net_file
        self.out_csv_name = out_csv_name

        self.env = gym.make(
                        env_name,
                        net_file=net_file,
                        route_file=route_file,
                        out_csv_name=out_csv_name,
                        use_gui=self.use_gui,
                        num_seconds=num_seconds
                    )
     
        self.state, _ = self.env.reset()
        self.done = False
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space
        self.traffic_signals = self.env.traffic_signals
    
    def reset(self):
        self.state, _ = self.env.reset()
        self.done = False
        return self.state
    
    def custom_reward(self, traffic_signal, reward_type='average_speed', reward_method='simple'):
        # print("Inside custom reward method")
        if reward_method == 'simple':
            match reward_type:
                case 'average_speed':
                    return traffic_signal.get_avgerage_speed()
                case 'congesion':
                    return -1 * traffic_signal.get_pressure()
                case 'emissions':
                    return -1* traffic_signal.get_emission_co2()
                case 'throughput':
                    return traffic_signal.get_throughput()

        else:
            # Weighted sum of the metrics
            reward = 0
            if weights is None:
                weights = {
                    'average_speed': 0.4,
                    'waiting_time': 0.3,
                    'emissions': 0.2,
                    'throughput': 0.1
                }

            # Calculate individual rewards
            average_speed = traffic_signal.get_average_speed()
            waiting_time = -1* traffic_signal._diff_waiting_time_reward()
            total_queue = -1 * traffic_signal.get_total_queued()
            congesion = traffic_signal.get_pressure()

            print(average_speed, waiting_time, total_queue, congesion)
            weighted_reward = (
                weights['average_speed'] * average_speed +
                weights['waiting_time'] * waiting_time +
                weights['emissions'] * total_queue +
                weights['throughput'] * congesion
            )

            return weighted_reward
            
            
    def step(self, action):
        # print("Inside step method")
        next_state, reward, terminated, truncated, info = self.env.step(action)
        # print("Step taken", next_state, terminated, truncated, info)

        # print("Traffic signals:", list(self.traffic_signals.values())[0])
        # traffic_signal = list(self.traffic_signals.values())[0]
        # print("Pressure:", traffic_signal.get_pressure())
        # # print(traffic_signal.get_average_speed(), traffic_signal.get_total_queued(), traffic_signal._diff_waiting_time_reward(), traffic_signal.get_pressure())
        # reward = self.custom_reward(traffic_signal, reward_type='congesion', reward_method='simple')
        # print("Reward:", reward)

        self.state = next_state
        self.done = terminated
        return next_state, reward, self.done or truncated
    
    def render(self):
        self.env.render()
    
    def close(self):
        try:
            self.env.close()
            if traci.isLoaded():
                traci.close()
            print("Env and Traci closed successfully.")
        except Exception as e:
            print("Error while closing the environment:", e)
    
    def get_state(self):
        return self.state


class SharedAdam(T.optim.Adam):
    def __init__(self, params, lr = 0.001, betas = (0.9, 0.99), eps = 1e-8, weight_decay = 0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = T.zeros_like(p.data)
                state['exp_avg_sq'] = T.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super(ActorCritic, self).__init__()

        # print("Inside ActorCritic class", input_dims, n_actions)

        self.gamma = gamma

        self.pi1 = nn.Linear(*input_dims, 128)
        self.v1 = nn.Linear(*input_dims, 128)
        self.pi = nn.Linear(128, n_actions)
        self.v = nn.Linear(128, 1)

        self.rewards = []
        self.actions = []
        self.states = []

    def store_in_mem(self, state, action, rewward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(rewward)

    def clear_mem(self):
        self.actions = []
        self.states = []
        self.rewards = []

    def forward(self, state):
        pi1 = F.relu(self.pi1(state))
        v1 = F.relu(self.v1(state))

        pi = self.pi(pi1)
        v = self.v(v1)

        return pi, v

    def calc_R(self, done):
        states = T.tensor(np.array(self.states), dtype=T.float)
        _, v = self.forward(states)

        R = v[-1] * (1-int(done))

        batch_return = []

        for reward in self.rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float)

        return batch_return


    
    def calc_loss(self, done):
        states = T.tensor(np.array(self.states), dtype=T.float)
        actions = T.tensor(self.actions, dtype=T.long)

        returns = self.calc_R(done)

        pi, values = self.forward(states)

        values = values.squeeze()
        critic_loss = (returns - values) ** 2

        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs * (returns - values)

        total_loss = (critic_loss + actor_loss).mean()

        return total_loss
    
    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float)
        pi, v = self.forward(state)
        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample().item()

        return action
    
    

class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions, gamma, lr, name, global_ep_idx, env_id, env,
                  n_episodes, rewards_list, C=5, grad_clip=5, num_seconds=1000, nets_file=None, routes_file=None, out_csv_name=None):
        super(Agent, self).__init__()

        self.env_id = env_id
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
        self.global_actor_critic = global_actor_critic

        self.name = "w%02i" % name
        self.episode_idx = global_ep_idx
        self.env_params = {
            'net_file': nets_file,
            'route_file': routes_file,
            'out_csv_name': out_csv_name,
            'render_mode': None,
            'num_seconds': num_seconds
        }

        self.optimizer = optimizer
        self.n_episodes = n_episodes
        self.rewards_list = rewards_list
        self.C = C
        self.grad_clip = grad_clip
        

    def run(self):
        t_step = 1
        max_steps = 5

        rewards_per_eps = {}

        self.env = Environment('sumo-rl-v0', **self.env_params)
        try:
            while self.episode_idx.value < self.n_episodes:
                terminated = False
                observation  = self.env.reset()

                score = 0
                self.local_actor_critic.clear_mem()

                while not terminated:
                    action = self.local_actor_critic.choose_action(observation)
                    obs, reward, terminated = self.env.step(action)
                    score += reward
                    self.local_actor_critic.store_in_mem(observation, action, reward)

                    if t_step % self.C == 0 or terminated:
                        loss = self.local_actor_critic.calc_loss(terminated)
                        self.optimizer.zero_grad()
                        loss.backward()

                        T.nn.utils.clip_grad_norm_(self.local_actor_critic.parameters(), self.grad_clip)

                        for local_param, global_param in zip(self.local_actor_critic.parameters(), self.global_actor_critic.parameters()):
                            global_param._grad = local_param.grad

                        self.optimizer.step()
                        self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())

                        self.local_actor_critic.clear_mem()
                    
                    t_step += 1
                    observation = obs

                with self.episode_idx.get_lock():
                    self.rewards_list.append(score)
                    self.episode_idx.value += 1
                
                print(self.name, "Episode", self.episode_idx.value, "reward %.1f" % score, flush=True)
        except Exception as e:
            print(f"Worker {self.name} encountered error: {e}")
        finally:
            if hasattr(self, 'env'):
                self.env.close()

def rewards_per_episode_plot_2(rewards_per_ep, environment_type, epsilons=None, window_size=10):
    """
    Plots rewards per episode with optional smoothing.
    
    Args:
        rewards_per_ep (dict): A dictionary where keys are episodes and values are rewards.
        environment_type (str): The type of environment (for plot title).
        epsilons (list, optional): Epsilon values per episode, if you'd like to include them in the plot. Default is None.
        window_size (int, optional): The window size for moving average smoothing. Default is 10.
    """
    episodes = np.arange(len(rewards_per_ep))
    rewards = np.array(rewards_per_ep).flatten()

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, marker='o', linestyle='-', markersize=4, label='Rewards per Episode')

    # Moving average for smoothing
    if window_size > 1:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(episodes[:len(moving_avg)], moving_avg, color='r', label=f'Moving Avg (window={window_size})')

    plt.title(f'Rewards per Episode {environment_type}', fontsize=20)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.grid(True)
    
    # Include epsilon plot if provided
    if epsilons:
        ax2 = plt.gca().twinx()
        ax2.plot(episodes, epsilons, color='g', alpha=0.6, linestyle='--', label='Epsilon')
        ax2.set_ylabel('Epsilon', fontsize=14)
        ax2.tick_params(axis='y', labelsize=12)

    plt.legend()
    plt.tight_layout()
    plt.show()
    # return plt

def train_a3c(env, input_files, env_id='CartPole-v1', input_dims=[4], n_actions=2, n_episodes=5000, gamma=0.99, use_wandb=False, grad_clip=0.5, C=5, lr=1e-4):
    print("Training started...", flush=True)
    
    # env_id = 'CartPole-v1'
    # gamma=0.99

    # env = gym.make(env_id)
    nets_file = input_files['nets_file']
    routes_file = input_files['routes_file']
    out_csv_name = input_files['out_csv_name']

    try:
        global_actor_critic = ActorCritic(input_dims, n_actions)
        global_actor_critic.share_memory()

        optim = SharedAdam(global_actor_critic.parameters(), lr=lr, betas=(0.92, 0.999))

        global_ep = mp.Value('i', 0)
        # global_ep_r = mp.Value('d', 0.)
        # result_queue = mp.Queue()

        thread_manager = mp.Manager()
        rewards_list = thread_manager.list()
        
        workers = [Agent(global_actor_critic, optim, input_dims, n_actions, gamma,
                        lr, i,
                        global_ep_idx=global_ep,
                        env_id=env_id,
                        env=None,
                        n_episodes=n_episodes, 
                        rewards_list=rewards_list, 
                        grad_clip=grad_clip, C=C, 
                        nets_file=nets_file, 
                        routes_file=routes_file, 
                        out_csv_name=out_csv_name) for i in range(mp.cpu_count())]

        [w.start() for w in workers]
        [w.join() for w in workers]


        folder_path = 'a3c_models'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        T.save(global_actor_critic, os.path.join(folder_path, f'a3c_model_{env_id}.pth' ))

        rewards_per_ep = list(rewards_list)
        rewards_per_episode_plot_2(rewards_per_ep=rewards_per_ep, environment_type=env_id)

        if use_wandb:
            wandb_config = {
                'env': env_id,
                'algorithm': 'A3C',
                'gamma': gamma,
                'lr': lr,
                'num_episodes': n_episodes,
                'input_dims': input_dims,
                'n_actions': n_actions
            }
            wandb.init(project='A3_A3C', config=wandb_config, name=env_id)

            for episode, reward in enumerate(rewards_list):
                wandb.log({'episode': episode, 'reward': reward})

            wandb.save(f'a3c_model_{env_id}.pth')

            wandb.finish()

        return rewards_list

    except KeyboardInterrupt:
        print("Training interrupted by user. Closing the environment.")
    
    except mp.TimeoutError:
        print("Multiprocessing timeout occurred. Training incomplete.")
    
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return []
    
    finally:
        try:
            env.close()
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
            thread_manager.shutdown()
            print("Environment and workers cleaned up successfully.")
        except Exception as e:
            print(f"Error during cleanup: {e}")

def greedy_agent_a3c(model_path, env_id, n_episodes=100):
    env = gym.make(env_id)
    # model = ActorCritic(env.observation_space.shape, env.action_space.n)
    # model.load_state_dict(T.load(model_path))
    # model.eval()
    mp.set_start_method('spawn')
    model = T.load(model_path)
    model.eval()


    reward_list = []
    rewards_per_episode = {}

    p_bar = tqdm(range(n_episodes), colour='red', desc='Testing Progress', unit='Episode')

    for episode in p_bar:
        observation, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            state = T.tensor([observation], dtype=T.float)
            with T.no_grad():
                pi, _ = model(state)
                action = T.argmax(pi).item()  # Choose the action with highest probability

            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        reward_list.append(episode_reward)
        rewards_per_episode[episode] = episode_reward

    print(rewards_per_episode)
    rewards_per_ep_array = np.array(list(rewards_per_episode.values())).flatten()
    avg_rewards_over_eps = np.mean(rewards_per_ep_array)

    return avg_rewards_over_eps, rewards_per_episode





if __name__ == "__main__":
    n_episodes = 700
    batch_size = 64
    epsilon = 1
    gamma = 0.99
    learning_rate = 1e-3
    epsilon_decay = 0.995

    C = 10
    num_seconds = 500

    nets_dir = 'nets'

    file_name = 'single_intersection_simple'
    env_id = f'sumo-rl-v0-{file_name}'


    nets_file = os.path.join(nets_dir, f'{file_name}.net.xml')
    routes_file = os.path.join(nets_dir, f'{file_name}.rou.xml')
    out_csv_name = os.path.join(nets_dir+"/a3c_results_csv", f'{file_name}.passenger.csv')

    file_exists = lambda file_path: os.path.exists(file_path)

    results_dir = os.path.join(nets_dir, 'a3c_results_csv')
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    if not file_exists(nets_file):
        raise FileNotFoundError(f"Net file not found: {nets_file}")
    if not file_exists(routes_file):
        raise FileNotFoundError(f"Route file not found: {routes_file}")

    sumo_env = Environment('sumo-rl-v0', net_file=nets_file, route_file=routes_file, out_csv_name=out_csv_name, render_mode=None, num_seconds=num_seconds)


    print("Observation Space:", sumo_env.observation_space)
    print("Action Space:", sumo_env.action_space)
    print("Initial State:", sumo_env.state)

    input_dims = [sumo_env.observation_space]
    n_actions = sumo_env.action_space.n

    global_actor_critic = ActorCritic(input_dims, n_actions)
    global_actor_critic.share_memory()

    optim = SharedAdam(global_actor_critic.parameters(), lr=learning_rate, betas=(0.92, 0.999))

    global_ep = mp.Value('i', 0)
    thread_manager = mp.Manager()
    rewards_list = thread_manager.list()

    input_files = {
        'nets_file': nets_file,
        'routes_file': routes_file,
        'out_csv_name': out_csv_name
    }

    rewards_per_episode = train_a3c(env=None, env_id=env_id, input_dims=input_dims, n_actions=n_actions, 
                                    n_episodes=n_episodes, gamma=gamma, use_wandb=False, grad_clip=0.5, C=5, lr=1e-4, input_files=input_files)

    # workers = [Agent(global_actor_critic, optim, input_dims, n_actions, gamma, env=sumo_env
    #                  learning_rate, i,
    #                  global_ep_idx=global_ep,
    #                  env_id=env_id,
    #                  n_episodes=n_episodes, rewards_list=rewards_list) for i in range(mp.cpu_count())]

    # [w.start() for w in workers]
    # [w.join() for w in workers]

    # # Save the model
    # T.save(global_actor_critic, f'a3c_model_{env_id}.pth')

    # rewards_per_ep = list(rewards_list)
    rewards_per_episode_plot_2(rewards_per_ep=rewards_per_episode, environment_type=env_id)



