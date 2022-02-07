import torch 
import copy
import random
from torch import nn 
from collections import deque

DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Memory:
    def __init__(self, exp_replay_size):
        self.experience = deque(maxlen = exp_replay_size)

    def collect(self, experience):
        self.experience.append(experience)

    def sample(self, sample_size):
        sample = random.sample(self.experience, sample_size)
        s = torch.tensor([exp[0] for exp in sample]).float()
        a = torch.tensor([exp[1] for exp in sample]).float()
        rn = torch.tensor([exp[2] for exp in sample]).float()
        sn = torch.tensor([exp[3] for exp in sample]).float()   
        return s, a, rn, sn

    def __len__(self):
        return len(self.experience)


class DQN_Agent:
    
    def __init__(self, env, seed, layer_sizes, lr, sync_freq, exp_replay_size):
        torch.manual_seed(seed)
        self.q_net = self.build_nn(layer_sizes)
        self.target_net = copy.deepcopy(self.q_net)
        self.q_net.to(DEV)
        self.target_net.to(DEV)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        
        self.network_sync_freq = sync_freq
        self.network_sync_counter = 0
        self.gamma = torch.tensor(0.95).float().to(DEV)
        self.memory = Memory(exp_replay_size)  
        
        self.env = env
        self.epsilon = 1
        
    def build_nn(self, layer_sizes):
        assert len(layer_sizes) > 1
        layers = []
        for index in range(len(layer_sizes)-1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index+1])
            act =    nn.Tanh() if index < len(layer_sizes)-2 else nn.Identity()
            layers += (linear,act)
        return nn.Sequential(*layers)
    
    def get_action(self, state):
        # We do not require gradient at this point, because this function will be used either
        # during experience collection or during inference
        with torch.no_grad():
            Qp = self.q_net(torch.from_numpy(state).float().to(DEV))
        _ , A = torch.max(Qp, axis=0)
        A = A if torch.rand(1,).item() > self.epsilon else torch.tensor(self.env.action_space.sample())
        return A
    
    def get_q_next(self, state):
        with torch.no_grad():
            qp = self.target_net(state)
        q,_ = torch.max(qp, axis=1)    
        return q
    
    def train(self, batch_size):
        s, a, rn, sn = self.memory.sample(sample_size=batch_size)
        if(self.network_sync_counter == self.network_sync_freq):
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.network_sync_counter = 0
        
        # predict expected return of current state using main network
        qp = self.q_net(s.to(DEV))
        pred_return, _ = torch.max(qp, axis=1)
        
        # get target return using target network
        q_next = self.get_q_next(sn.to(DEV))
        target_return = rn.to(DEV) + self.gamma * q_next
        
        loss = self.loss_fn(pred_return, target_return)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        
        self.network_sync_counter += 1       
        return loss.item()