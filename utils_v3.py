import torch 
import numpy as np
from torch import nn
from time import sleep
from gym.wrappers import Monitor

exp_replay_size = 256

# CONVERTS TRAJ LIST TO STEP LIST
def preprocess_traj(traj_list, step_list, is_Demo=False):
    step_list = step_list.tolist()
    for traj in traj_list:
        states = np.array(traj[0])
        if is_Demo:
            probs = np.ones((states.shape[0], 1))
        else:
            probs = np.array(traj[1]).reshape(-1, 1)
        actions = np.array(traj[2]).reshape(-1, 1)
        x = np.concatenate((states, probs, actions), axis=1)
        step_list.extend(x)
    return np.array(step_list)


def init_experiance_replay(agent, env):
    # initiliaze experiance replay      
    for i in range(exp_replay_size):
        obs = env.reset()
        done = False
        while(done != True):
            A = agent.get_action(obs, env.action_space.n, epsilon=1)
            obs_next, reward, done, _ = env.step(A.item())
            agent.collect_experience([obs, A.item(), reward, obs_next])
            obs = obs_next
            if(len(agent.memory) > exp_replay_size):
                break


def generate_session(agent, env, cost_f=None, train=False, batch_size=16):
    states, actions, rewards= [], [], [] 
    state = env.reset()
    done = False
    #episode_len = 0
    #episode_loss = 0 
    while not done:
        #episode_len += 1 
        A = agent.get_action(state)
        next_state, reward, done, _ = env.step(A.item())
        if isinstance(cost_f, nn.Module):
            state_torch = torch.tensor(state, dtype=torch.float32)
            action_torch = torch.tensor(A.item(), dtype=torch.float32)
            x = torch.cat((state_torch.reshape(-1, 1), action_torch.reshape(-1, 1)))
            y = torch.transpose(x, 0, 1)
            cost = cost_f(y).detach().numpy()
            agent.memory.collect([state, A.item(), -cost, next_state])
        else:
            agent.memory.collect([state, A.item(), reward, next_state])

        state = next_state
        states.append(state)
        actions.append(A.item())
        rewards.append(reward)
        
        if(len(agent.memory) > 128):
            if train: 
                for _ in range(4):
                    loss = agent.train(batch_size=batch_size)
                    #episode_loss += loss

    if agent.epsilon > 0.05 :
        agent.epsilon -= (1 / 5000)
    
    return states, actions, rewards


def render(agent, env):
    env = Monitor(env, './video', force=True)
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state).item()
        state, reward, done, info = env.step(action)
        sleep(0.01)
        env.render()  
    env.close()
    