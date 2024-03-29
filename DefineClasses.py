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

        # Convolutions
        self.num_kernels1 = 16
        self.conv1 = nn.Conv2d(4, self.num_kernels1, kernel_size=8, stride=4, padding=1)
        self.num_kernels2 = 32
        self.conv2 = nn.Conv2d(self.num_kernels1, self.num_kernels2, kernel_size=4, stride=2, padding=1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=self.num_kernels2 * 10 * 10, out_features=256)
        self.out = nn.Linear(in_features=256, out_features=num_actions)

    def forward(self, t):
        t = t.to(self.device)
        t = F.relu(self.conv1(t))
        t = F.relu(self.conv2(t))
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = self.out(t)
        return t.cpu()


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

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return random.sample(self.memory, batch_size)

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
        self.current_steps = []
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device
        self.memory = memory

    def select_action(self, state, policy_net, optimal=False, timestep=0):
        if optimal:
            return self.select_optimal_action(state, policy_net)

        if timestep >= len(self.current_steps):
            self.current_steps = np.append(self.current_steps, 0)

        rate = self.strategy.get_exploration_rate(self.current_steps[timestep])
        if self.memory.can_provide_sample():
            self.current_steps[:timestep] += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)  # explore
        else:
            return self.select_optimal_action(state, policy_net)  # exploit

    def select_optimal_action(self, state, policy_net):
        with torch.no_grad():
            return policy_net(state).argmax(dim=1).to(self.device)


class EnvManager:
    def __init__(self, environment, device, k=4):
        self.device = device
        self.env = gym.make(environment).unwrapped
        self.env.reset()
        self.k = k
        self.stack_screens = []
        self.done = False

    def reset(self):
        self.env.reset()
        self.stack_screens = deque(maxlen=4)

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def num_actions_available(self):
        return self.env.action_space.n

    def take_action(self, action):
        _, reward, self.done, _ = self.env.step(action)
        rewards = reward
        # Skip frames and take the same actions for those being skipped (left or right)
        for _ in range(self.k - 1):
            if action == 1:  # action Fire should just pick Noop
                _, reward, self.done, _ = self.env.step(0)
            else:
                _, reward, self.done, _ = self.env.step(action)
            rewards += reward
            if self.done:
                break
        return torch.tensor([rewards])

    def just_starting(self):
        return len(self.stack_screens) < 4

    def get_state(self):
        if self.done:
            self.stack_screens.append(self.get_processed_screen())
            black_screen = torch.zeros((1, 4, self.get_screen_width(), self.get_screen_height()))
            return black_screen
        else:
            if self.just_starting():
                self.stack_screens.append(self.get_processed_screen())
                self.stack_screens.append(self.get_processed_screen())
                self.stack_screens.append(self.get_processed_screen())
                self.stack_screens.append(self.get_processed_screen())

            result_screen = torch.zeros((4, self.get_screen_width(), self.get_screen_height()))
            for i in range(len(self.stack_screens)):
                result_screen[i] = self.stack_screens[i]

            self.stack_screens.append(self.get_processed_screen())
            return result_screen.unsqueeze(0)

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
