import random
from collections import deque

import gym
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import matplotlib.pyplot as plt


class DQN(nn.Module):
    def __init__(self, num_actions, device='cpu'):
        super().__init__()
        self.device = device
        img_height = 84
        img_width = 84
        #Convolutions
        self.num_kernels1 = 32
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=self.num_kernels1, kernel_size=3, stride=1, padding=1) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.maxpool_output_size1 = int(self.num_kernels1 *int(img_height / 2) * int(img_width / 2))

        #Fully Connected Layers with max pool
        # self.fc1 = nn.Linear(in_features=self.maxpool_output_size1, out_features=24)
        # self.out = nn.Linear(in_features=24, out_features=num_actions)

        # #Fully Connected Layers with convoltional
        # self.fc1 = nn.Linear(in_features=self.num_kernels1*img_height*img_width, out_features=24)   
        # self.out = nn.Linear(in_features=24, out_features=num_actions)

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=1*img_height*img_width, out_features=24)
        self.out = nn.Linear(in_features=24, out_features=num_actions)



    def forward(self, t):
        t = t.to(self.device)

        # t = self.conv1(t)
        # t = self.pool1(t)
        # t= F.relu(t)

        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = self.out(t)
        return t
        
       


class ReplayMemory:
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
        self.batch_size = batch_size

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count] = experience
        self.push_count = (self.push_count + 1) % self.capacity

    def sample(self, percentage=1):
        return random.sample(self.memory, int(self.batch_size*percentage))

    def can_provide_sample(self):
        return len(self.memory) >= self.batch_size


class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
               math.exp(-1. * current_step * self.decay)


class Agent:
    def __init__(self, strategy, num_actions, device, memory):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device
        self.memory = memory

    def select_action(self, state, policy_net, optimal=False):
        if optimal:
            return self.select_optimal_action(state, policy_net)

        rate = self.strategy.get_exploration_rate(self.current_step)
        if self.memory.can_provide_sample():
            self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)  # explore
        else:
            return self.select_optimal_action(state, policy_net)  # exploit

    def select_optimal_action(self, state, policy_net):
        with torch.no_grad():
            return policy_net(state).argmax(dim=1).to(self.device)


class EnvManager():
    def __init__(self, environment, device, k=4):
        self.device = device
        self.env = gym.make(environment).unwrapped
        self.env.reset()
        self.k = k
        self.stack_screens = []
        self.done = False

    def reset(self):
        self.env.reset()
        self.stack_screens = deque(maxlen=self.k)

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def num_actions_available(self):
        return self.env.action_space.n

    def take_action(self, action):
        _, reward, self.done, _ = self.env.step(action.item())
        rewards = [reward]
        # Skip frames and take the same actions for those being skipped (left or right)
        for _ in range(self.k - 1):
            if action.item() == 1:  # action Fire should just pick Noop
                _, reward, self.done, _ = self.env.step(0)
            else:
                _, reward, self.done, _ = self.env.step(action.item())
            rewards.append(reward)
            if self.done:
                break
        return torch.tensor([np.asarray(rewards).sum()])

    def just_starting(self):
        return len(self.stack_screens) < self.k

    def get_state(self):
        if self.just_starting() or self.done:
            # Produce a stack of 4 black screens
            self.stack_screens.append(self.get_processed_screen())
            black_screen = torch.zeros((1, 1, self.get_screen_width(), self.get_screen_height()))
            return black_screen
        else:
            # Stack four frames - 4 channel
            # result_screen = torch.zeros((self.k, self.get_screen_width(), self.get_screen_height()))
            # for i in range(len(self.stack_screens)):
            #     result_screen[i] = self.stack_screens[i]

            #Screen substraction test - 1 channel
            result = self.stack_screens[0]
            for i in range(1,len(self.stack_screens)):
                result -= self.stack_screens[i]

            self.stack_screens.append(self.get_processed_screen())
            return result.unsqueeze(0)

            
            # return result_screen.unsqueeze(0)

    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2, 0, 1))  # PyTorch expects CHW
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):
        # Strip off top and bottom
        top = 30
        bottom = 195
        screen = screen[:, top:bottom, :]
        return screen

    def transform_screen_data(self, screen):
        # Convert to float, rescale, convert to tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32)
        screen = torch.from_numpy(screen)

        # Use torchvision package to compose image transforms
        resize = T.Compose([
            T.ToPILImage()
            , T.transforms.Grayscale()
            , T.Resize((84, 84))
            , T.ToTensor()
        ])

        return resize(screen)  # add a batch dimension (BCHW)

    def plot_screen(self, screen=None):
        """
        Plot the provided screen; otherwise plot the current processed screen
        :param screen:
        :return:
        """
        if screen is None:
            screen = self.get_processed_screen().cpu()
        screen = screen.resize_(84, 84)
        plt.imshow(screen)
        plt.show()

    def get_screen_height(self):
        if len(self.stack_screens) == 0:
            screen = self.get_processed_screen()
            return screen.shape[1]
        else:
            return self.stack_screens[0].shape[1]

    def get_screen_width(self):
        if len(self.stack_screens) == 0:
            screen = self.get_processed_screen()
            return screen.shape[2]
        else:
            return self.stack_screens[0].shape[2]
