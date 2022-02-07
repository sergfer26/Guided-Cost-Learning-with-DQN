from tqdm import tqdm 
import numpy as np
import torch 

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


def generate_session(agent, env, noise, cost, train=False, steps=1000, batch_size=128):
    states, actions, rewards = [], [], []
    state = env.reset()
    noise.reset()
    episode_reward = 0
    for step in range(steps):
        action = agent.get_action(state)
        #action = noise.get_action(action, step)
        other = 1 - action
        action_probs = np.array([action, other])
        a = np.random.choice(env.action_space.n,  p=action_probs)
        new_state, reward, done, _ = env.step(a)
        ##################### GCL #####################
        state_torch = torch.tensor(state, dtype=torch.float32)
        action_torch = torch.tensor(a, dtype=torch.float32)
        x = torch.cat((state_torch.reshape(-1, 1), action_torch.reshape(-1, 1)))
        y = torch.transpose(x, 0, 1)
        new_reward = cost(y).detach().numpy()
        ##################### GCL #####################
        agent.memory.push(state, a, -new_reward, new_state, done)
        if train:
            if len(agent.memory) > batch_size:
                agent.update(batch_size)

        states.append(state)
        actions.append(a)
        rewards.append(reward) 

        state = new_state
        episode_reward += reward
        if done:
            #sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
            break
        #rewards.append(episode_reward)
        #avg_rewards.append(np.mean(rewards[-10:]))
    return states, actions, rewards