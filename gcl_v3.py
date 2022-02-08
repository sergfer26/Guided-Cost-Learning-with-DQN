import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

from DQN.dqn import DQN_Agent 
from GCL.cost import CostNN
#from GCL.utils import to_one_hot, get_cumulative_rewards

#from torch.optim.lr_scheduler import StepLR
from mail import send_mail
from utils_v3 import generate_session, preprocess_traj, render, exp_replay_size

# SEEDS
seed = 18095048
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ENV SETUP
env_name = 'CartPole-v0'
env = gym.make(env_name).unwrapped
if seed is not None:
    env.seed(seed)
n_actions = env.action_space.n
state_shape = env.observation_space.shape
state = env.reset()

# LOADING EXPERT/DEMO SAMPLES
demo_trajs = np.load('GCL/expert_samples/pg_cartpole.npy', allow_pickle=True)
print(len(demo_trajs))

# INITILIZING POLICY AND REWARD FUNCTION
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
agent = DQN_Agent(env, seed=seed, layer_sizes=[input_dim, 64, output_dim], lr=1e-3, sync_freq=5, exp_replay_size=exp_replay_size)

cost_f = CostNN(state_shape[0] + 1)
cost_optimizer = torch.optim.Adam(cost_f.parameters(), 1e-2, weight_decay=1e-4)

mean_rewards = []
mean_costs = []
mean_loss_rew = []
mean_rewards_ = []
mean_costs_ = []
EPISODES_TO_PLAY = 1
EPISODES = 40
REWARD_FUNCTION_UPDATE = 10
DEMO_BATCH = 100
sample_trajs = []

D_demo, D_samp = np.array([]), np.array([])

D_demo = preprocess_traj(demo_trajs, D_demo, is_Demo=True)
return_list, sum_of_cost_list = [], []
return_list_, sum_of_cost_list_ = [], []
for i in range(EPISODES):
    trajs = [generate_session(agent, env, cost_f) for _ in range(EPISODES_TO_PLAY)]
    sample_trajs = trajs + sample_trajs
    D_samp = preprocess_traj(trajs, D_samp)

    # UPDATING REWARD FUNCTION (TAKES IN D_samp, D_demo)
    loss_rew = []
    for _ in range(REWARD_FUNCTION_UPDATE):
        selected_samp = np.random.choice(len(D_samp), DEMO_BATCH)
        selected_demo = np.random.choice(len(D_demo), DEMO_BATCH)

        D_s_samp = D_samp[selected_samp]
        D_s_demo = D_demo[selected_demo]

        #D̂ samp ← D̂ demo ∪ D̂ samp
        D_s_samp = np.concatenate((D_s_demo, D_s_samp), axis = 0)

        states, probs, actions = D_s_samp[:,:-2], D_s_samp[:,-2], D_s_samp[:,-1]
        states_expert, actions_expert = D_s_demo[:,:-2], D_s_demo[:,-1]

        # Reducing from float64 to float32 for making computaton faster
        states = torch.tensor(states, dtype=torch.float32)
        probs = torch.tensor(probs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        states_expert = torch.tensor(states_expert, dtype=torch.float32)
        actions_expert = torch.tensor(actions_expert, dtype=torch.float32)

        costs_samp = cost_f(torch.cat((states, actions.reshape(-1, 1)), dim=-1))
        costs_demo = cost_f(torch.cat((states_expert, actions_expert.reshape(-1, 1)), dim=-1))

        # LOSS CALCULATION FOR IOC (COST FUNCTION)
        loss_IOC = torch.mean(costs_demo) + \
                torch.log(torch.mean(torch.exp(-costs_samp)/(probs+1e-7)))
        # UPDATING THE COST FUNCTION
        cost_optimizer.zero_grad()
        loss_IOC.backward()
        cost_optimizer.step()

        loss_rew.append(loss_IOC.detach())

    for traj in trajs:
        states, actions, rewards = traj
        states = torch.tensor(states, dtype=torch.float32)
        #probs = torch.tensor(probs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        costs = cost_f(torch.cat((states, actions.reshape(-1, 1)), dim=-1)).detach().numpy()
    
    returns = sum(rewards)
    sum_of_cost = np.sum(costs)
    return_list.append(returns)
    sum_of_cost_list.append(sum_of_cost)

    mean_rewards.append(np.mean(return_list))
    mean_costs.append(np.mean(sum_of_cost_list))
    mean_loss_rew.append(np.mean(loss_rew))

    # TRAINING ACTOR-CRITIC NETWORKS 
    trajs_ = [generate_session(agent, env, cost_f, train=True) for _ in range(EPISODES_TO_PLAY)]
    for traj in trajs_:
        states, actions, rewards = traj
        states = torch.tensor(states, dtype=torch.float32)
        #probs = torch.tensor(probs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        costs = cost_f(torch.cat((states, actions.reshape(-1, 1)), dim=-1)).detach().numpy()

    # PLOTTING PERFORMANCE
    if i % 10 == 0:
        # clear_output(True)
        print(f"mean reward:{np.mean(return_list)} loss: {loss_IOC}")

        plt.figure(figsize=[16, 12])
        plt.subplot(2, 2, 1)
        plt.title(f"Mean reward per {EPISODES_TO_PLAY} games")
        plt.plot(mean_rewards)
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.title(f"Mean cost per {EPISODES_TO_PLAY} games")
        plt.plot(mean_costs)
        plt.grid()

        plt.subplot(2, 2, 3)
        plt.title(f"Mean loss per {REWARD_FUNCTION_UPDATE} batches")
        plt.plot(mean_loss_rew)
        plt.grid()

        # plt.show()
        plt.savefig('plots/GCL_learning_curve_v3.png')
        plt.close()

    if np.mean(return_list) > EPISODES:
        break

#render(agent, env)
agent.save('/net')
send_mail('plots/GCL_learning_curve_v3.png')
