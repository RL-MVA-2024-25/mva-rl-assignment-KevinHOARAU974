from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from fast_env import FastHIVPatient
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from tqdm import tqdm
import pickle
import random
import torch
import matplotlib.pyplot as plt
from copy import deepcopy

env = TimeLimit(
    env=FastHIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

#First strategy
class ProjectAgent:
    
    # def __init__(self):
        
    #     self.DQN
        
    #     self.nb_neurons
    #     self.nb_actions
    #     self.state_dimension
        
    #     self.device
    #     self.gamma
    #     self.batch_size
    #     self.memory
    #     self.epsilon_max
    #     self.epsilon_min
    #     self.epsilon_stop
    #     self.epsilon_delay
    #     self.epsilon_step
        
    #     self.loss
    #     self.optimizer
        
    def act(self, observation, use_random=False):
        with torch.no_grad():
            Q = self.DQN(torch.Tensor(observation).unsqueeze(0).to(device=self.device))
            return torch.argmax(Q).item()

    def save(self, path):
        
        data_to_save = {"nb_actions": self.nb_actions,
                        "state_dimension": self.state_dimension,
                        "nb_neurons": self.nb_neurons,
                        "param_nn": self.DQN.state_dict()}
        
        with open(path, 'wb') as f:
            pickle.dump(data_to_save, f)
        

    def load(self):
        path = "DQN_8.pkl"
        
        with open(path, 'rb') as f:
            loaded_data = pickle.load(f)
            
        self.nb_actions = loaded_data["nb_actions"]
        self.state_dimension = loaded_data["state_dimension"]
        self.nb_neurons = loaded_data["nb_neurons"]
        
        param_nn = loaded_data["param_nn"]
        self.DQN = NeuralNetwork(self.state_dimension, self.nb_neurons, self.nb_actions)
        self.DQN.load_state_dict(param_nn)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.DQN.to(device=self.device)
    
    def set_config(self,config):
        
        self.device = config["device"]
        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]
        self.buffer_size = config["buffer_size"]
        self.memory = ReplayBuffer(self.buffer_size, self.device)
        
        self.epsilon_max = config["epsilon_max"]
        self.epsilon_min = config["epsilon_min"]
        self.epsilon_stop = config["epsilon_decay_period"]
        self.epsilon_delay = config["epsilon_delay_decay"]
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        
        self.state_dimension = config["state_dimension"]
        self.nb_neurons = config["nb_neurons"]
        self.nb_actions = config["nb_actions"]
        
        self.nb_gradient_steps = config['gradient_steps']
        self.update_target_strategy = config['update_target_stategy']
        self.update_target_freq = config['update_target_freq']
        self.update_target_tau = config['update_target_tau']
        
        self.DQN = NeuralNetwork(self.state_dimension, self.nb_neurons, self.nb_actions).to(self.device)
        self.target_network = deepcopy(self.DQN).to(self.device)
        
        self.loss = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.DQN.parameters(), lr=config["learning_rate"])
        
    def gradient_step(self):
        
        if len(self.memory) > self.batch_size:
            
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_network(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.DQN(X).gather(1, A.to(torch.long).unsqueeze(1))
            l = self.loss(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.DQN.parameters(), max_norm=0.5)
            self.optimizer.step()
            
    def fill_replay_buffer(self,env):
        
        print("Start filling")
        state,_ = env.reset()
        for _ in range(self.buffer_size):
            
            action = env.action_space.sample()
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            
            if done or trunc:
                state,_ = env.reset()
            else:
                state = next_state
        print("filling done")
        
    def train(self, env, max_episode):
        
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state,_ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        
        best_cum_reward =0
        
        while episode < max_episode:
            
            #update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            
            #select epsilon-greedy action 
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.act(state)
                
            #step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            
            #train
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()
                
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0:
                    self.target_network.load_state_dict(self.DQN.state_dict())
                    
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_network.state_dict()
                model_state_dict = self.DQN.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_network.load_state_dict(target_state_dict)
            
            #next transition
            step += 1
            # print(f"step : {step}")
            if done or trunc:
                
                episode += 1
                #print(f"Episode : {episode},\n epsilon : {epsilon},\n batch_size : {len(self.memory)},\n episode_return : {episode_cum_reward}")
                state,_ = env.reset()
                episode_return.append(episode_cum_reward)
                # if episode_cum_reward > best_cum_reward:
                #     # self.save("DQN_6.pkl")
                #     best_cum_reward = episode_cum_reward
                episode_cum_reward = 0
                
            else:
                state = next_state
        self.save("DQN_9.pkl")
        return episode_return
                
class NeuralNetwork(torch.nn.Module):
    
    def __init__(self, state_dim, nb_neurons, nb_action):
        
        
        super().__init__()
        self.DQN = torch.nn.Sequential(torch.nn.Linear(state_dim, nb_neurons),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(nb_neurons, nb_neurons),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(nb_neurons, nb_neurons),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(nb_neurons, nb_neurons),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(nb_neurons, nb_neurons),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(nb_neurons, nb_action))
    
    def forward(self,x):
        return self.DQN(x)
    
    def __call__(self,x):
        return self.forward(x)
    
class ReplayBuffer:
    
    def __init__(self, capacity, device):
        
        self.capacity = capacity #capacity of the buffer
        self.data = []
        self.index = 0 #index of the next cell to be filled
        self.device = device
        
    def append(self, s, a, r, s2, d):
        
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s2, d)
        self.index = (self.index +1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x: torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
        
    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    print('DQN Training')
    print(f"Observation space : {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    print("Start training")
    agent = ProjectAgent()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(device)
    
    state_dim = env.observation_space.shape[0]
    nb_actions = env.action_space.n
    nb_neurons = 128
    
    config = {"device" : device,
              "gamma" : 0.95,
              "batch_size" : 1024,
              "buffer_size" : 100000,
              "epsilon_min" : 0.05,
              "epsilon_max" : 0.9,
              "epsilon_decay_period" : 500000,
              "epsilon_delay_decay" : 10000,
              "delay_update_buffer" : 50000,
              "state_dimension" : state_dim,
              "nb_actions" : nb_actions,
              "nb_neurons" : nb_neurons,
              "learning_rate" : 0.00005,
              'gradient_steps' : 1,
              'update_target_stategy' : "ema",
              'update_target_freq' : 20,
              'update_target_tau' : 0.001}
    
    
    agent = ProjectAgent()
    agent.set_config(config)
    agent.fill_replay_buffer(env)
    
    scores = agent.train(env, 50000)
    # plt.plot(scores)            
    
    def smooth(data, window=50):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    plt.plot(smooth(scores))
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.title("Training Performance")
    plt.show()

