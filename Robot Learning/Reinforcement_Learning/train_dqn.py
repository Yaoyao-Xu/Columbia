import argparse
import numpy as np
import time
import random
import os
import signal
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.optim as optim
import time

from replay_buffer import ReplayBuffer
from q_network import QNetwork
from arm_env import ArmEnv


#---------- Utils for setting time constraints -----#
class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
#---------- End of timing utils -----------#


class TrainDQN:

    @staticmethod
    def add_arguments(parser):
        # Common arguments
        parser.add_argument('--learning_rate', type=float, default=7e-4,
                            help='the learning rate of the optimizer')
        # LEAVE AS DEFAULT THE SEED YOU WANT TO BE GRADED WITH
        parser.add_argument('--seed', type=int, default=45,
                            help='seed of the experiment')
        parser.add_argument('--save_dir', type=str, default='models',
                            help="the root folder for saving the checkpoints")
        parser.add_argument('--gui', action='store_true', default=False,
                            help="whether to turn on GUI or not")
        # 7 minutes by default
        parser.add_argument('--time_limit', type=int, default=7*60,
                            help='time limits for running the training in seconds')


    def __init__(self, env, device, args):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = False
        self.env = env
        self.env.seed(args.seed)
        self.env.observation_space.seed(args.seed)
        self.device = device
        # print(self.device.__repr__())
        # print(self.q_network)




        ## Hyper_parameter
        self.pre_time = 100
        self.action_num = 37
        self.epsilon_a = 0.05
        self.epsilon_b = 0.95
        self.episode = 217
        self.prepare = 10
        self.step_size = 200
        self.batch_size = 100
        self.update = 10
        self.criterion = nn.MSELoss()
        self.LR = args.learning_rate 
        self.discount = 0.999
        self.memory_size = 4000
        self.max_norm = 1.0
        


        self.memory = ReplayBuffer(self.memory_size)
        self.q_network = QNetwork(env).to(device)
        self.t_network = QNetwork(env).to(device)
        self.t_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.LR)
        self.q_network.train()
        self.t_network.eval()
    
        

        ##



    def save_model(self, episode_num, episode_reward, args):
        model_folder_name = f'episode_{episode_num:06d}_reward_{round(episode_reward):03d}'
        if not os.path.exists(os.path.join(args.save_dir, model_folder_name)):
            os.makedirs(os.path.join(args.save_dir, model_folder_name))
        torch.save(self.q_network.state_dict(), os.path.join(args.save_dir, model_folder_name, 'q_network.pth'))
        print(f'model saved to {os.path.join(args.save_dir, model_folder_name, "q_network.pth")}\n')
                
    
    def get_experience(self): # get experience for sampling 
        for i in range(self.prepare):
            s_t = self.env.reset()
            for j in range(self.pre_time):
                dis_action = np.random.randint(self.action_num)
                continuous_action = self.q_network.action_discrete_to_continuous(dis_action)
                s_t1, reward, done, info = self.env.step(continuous_action)
                transition = (s_t, dis_action, reward, s_t1, done)
                self.memory.put(transition)
                s_t= s_t1
    
    
    def train(self, args):
        #--------- YOUR CODE HERE --------------
        self.get_experience()
        

        # Deep Q-learning
        for episode_num in range(self.episode):
            episode_reward = 0
            state = self.env.reset()
            
            while 1:
                # exponential decay of epsilon
                grad = np.exp(-1. * self.env.num_steps / self.step_size)
                epsilon = (self.epsilon_a - self.epsilon_b) * grad +self.epsilon_b
                
                if np.random.random() <= epsilon:
                    #print(np.random.random())
                    discrete_action = np.random.randint(self.action_num)
                else:
                    discrete_action = self.q_network.select_discrete_action(state, self.device)

                # Store transition
                continuous_action = self.q_network.action_discrete_to_continuous(discrete_action)
                observation, reward, done, _ = self.env.step(continuous_action)
                if done == True:
                    break
                episode_reward += reward
                transition = (state, discrete_action, reward, observation, done)
                self.memory.put(transition)
            
                # train Network

                cur_state, cur_action, re, next_state, _ = self.memory.sample(self.batch_size)
                cur_action = torch.LongTensor(cur_action.reshape(-1,1))
                re = torch.FloatTensor(re)
                q_a = self.q_network.forward(cur_state, self.device).gather(1, cur_action)
                q_t = self.t_network.forward(next_state, self.device).max(dim = 1)[0].detach()
                learned = re + self.discount*q_t
            

                # back_prob
                self.optimizer.zero_grad()
                Loss = self.criterion(q_a, learned.unsqueeze(1))
                Loss.backward()
                nn.utils.clip_grad_norm_(list(self.q_network.parameters()), self.max_norm)
                self.optimizer.step()

                state = observation

            
            self.save_model(episode_num, episode_reward, args)
            if episode_num % self.update ==0:
                self.t_network.load_state_dict(self.q_network.state_dict())

      
        #---------------------------------------        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    TrainDQN.add_arguments(parser)
    args = parser.parse_args()
    args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.save_dir = os.path.join(args.save_dir, args.timestr)
    if not args.seed: args.seed = int(time.time())

    env = ArmEnv(args)
    device = torch.device('cpu')
    # declare Q function network and set seed
    tdqn = TrainDQN(env, device, args)
    # run training under time limit
    try:
        with time_limit(args.time_limit):
            tdqn.train(args)
    except TimeoutException as e:
        print("You ran out of time and your training is stopped!")
    
