import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

# Memory Replay Experience
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)


class DQN(nn.Module):
    def __init__(self, img_height, img_width, num_actions):
        super().__init__()

        # Convolutions
        self.num_kernels1 = 16
        self.conv1 = nn.Conv2d(4, self.num_kernels1, kernel_size=8, stride=4, padding=1)
        self.num_kernels2 = 32
        self.conv2 = nn.Conv2d(self.num_kernels1, self.num_kernels2, kernel_size=4, stride=2, padding=1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=self.num_kernels2 * 10 * 10, out_features=256)
        self.out = nn.Linear(in_features=256, out_features=num_actions)

    def forward(self, t):
        t = t.to(device)
        t = F.relu(self.conv1(t))
        t = F.relu(self.conv2(t))
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = self.out(t)
        return t.cpu()


class ReplayMemory():
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
        self.batch_size = batch_size

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def can_provide_sample(self):
        return len(self.memory) >= self.batch_size


class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
               math.exp(-1. * current_step * self.decay)


class Agent():
    def __init__(self, strategy, num_actions, device, memory):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device
        self.memory = memory

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        if self.memory.can_provide_sample():
            self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)  # explore
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device)  # exploit

    def select_optimal_action(self, state, policy_net):
        return policy_net(state).argmax(dim=1).to(self.device)


class EnvManager():
    def __init__(self, environment, device):
        self.device = device
        self.env = gym.make(environment).unwrapped
        self.env.reset()
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
        _, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward])

    def just_starting(self):
        return len(self.stack_screens) < 4

    def get_state(self, skip_frame=False):
        if self.just_starting() or self.done:
            self.stack_screens.append(self.get_processed_screen())
            black_screen = torch.zeros((1, 4, self.get_screen_width(), self.get_screen_height()))
            return black_screen
        else:
            result_screen = torch.zeros((4, self.get_screen_width(), self.get_screen_height()))
            for i in range(len(self.stack_screens)):
                result_screen[i] = self.stack_screens[i]
            if not skip_frame:
                self.stack_screens.append(self.get_processed_screen())
            return result_screen.unsqueeze(0)

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


def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Mean Score')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    plt.pause(0.001)
    print("Episode", len(values), "\n", moving_avg_period, "episode moving avg:", moving_avg[-1])
    if is_ipython: display.clear_output(wait=True)


def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period - 1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()


def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return t1, t2, t3, t4


class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).to(device).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size) - 10
        if len(non_final_states) > 0:
            with torch.no_grad():
                values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0]
        return values.to(QValues.device)


if __name__ == "__main__":
    environment = 'Breakout-v0'
    batch_size = 512
    gamma = 0.85
    eps_start = 1.0
    eps_end = 0.1
    eps_decay = 0.0001
    target_update = 2
    memory_size = 100000  # 11377 cuda 66000 cpu
    lr = 0.001
    num_episodes = 500
    k = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    em = EnvManager(environment, device)
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
    memory = ReplayMemory(memory_size, batch_size)
    agent = Agent(strategy, em.num_actions_available(), device, memory)

    policy_net = DQN(em.get_screen_height(), em.get_screen_width(), em.num_actions_available()).to(device)
    target_net = DQN(em.get_screen_height(), em.get_screen_width(), em.num_actions_available()).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

    episode_durations = []

    # screen = em.get_processed_screen().cpu()
    # screen = screen.resize_(84, 84)
    # plt.imshow(screen)
    # plt.show()

    policy_net.load_state_dict(torch.load("breakout-v0-1c.pth"))
    for episode in range(num_episodes):
        em.reset()
        state = em.get_state()

        scores = []
        for timestep in count():
            # screen = em.get_processed_screen().cpu()
            # screen = screen.resize_(84, 84)
            # plt.imshow(screen)
            # plt.show()

            if timestep % k == 0:
                action = agent.select_action(state, policy_net)
                reward = em.take_action(action)
                if reward.eq(0):
                    reward -= 0.0001
                elif reward.eq(1):
                    reward *= 10
                scores.append(reward.item())
                next_state = em.get_state()
                memory.push(Experience(state, action, next_state, reward))
            else:
                action = agent.select_optimal_action(state, policy_net)
                next_state = em.get_state(skip_frame=True)
            state = next_state

            loss = None
            if memory.can_provide_sample():
                experiences = memory.sample()
                states, actions, rewards, next_states = extract_tensors(experiences)

                current_q_values = QValues.get_current(policy_net, states, actions)
                next_q_values = QValues.get_next(target_net, next_states)
                target_q_values = (next_q_values * gamma) + rewards.to(device)

                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                del current_q_values
                del next_q_values
                torch.cuda.empty_cache()

            if em.done:
                scores = np.asarray(scores)
                episode_durations.append(scores.mean())
                plot(episode_durations, 5)
                print(" Scores array", scores[scores == 10])
                print(" Epsilon", strategy.get_exploration_rate(agent.current_step))
                print(" Memory len", memory.push_count)
                if loss:
                    print(" Loss", loss.item())
                break

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        torch.save(policy_net.state_dict(), 'breakout-v0-1c.pth')

    policy_net.eval()
    for episode in range(20):
        em.reset()
        state = em.get_state()

        for t in range(2000):
            em.render()
            action = agent.select_optimal_action(state, policy_net)
            reward = em.take_action(action)
            state = em.get_state()
            if em.done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    em.close()
