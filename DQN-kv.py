import time
from itertools import count

import torch.optim as optim

from DefineClasses import *
from utils import *

if __name__ == "__main__":
    environment = 'Breakout-v0'
    batch_size = 256
    gamma = 0.99
    eps_start = 0.1
    eps_end = 0.1
    eps_decay = 0.001
    target_update = 5
    memory_size = 100000
    lr = 0.001
    num_episodes = 500
    k = 1  # of skipped frames

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    em = EnvManager(environment, device, k=k)
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
    memory = ReplayMemory(memory_size, batch_size)
    agent = Agent(strategy, em.num_actions_available(), device, memory)

    policy_net = DQN(em.num_actions_available(), device).to(device)
    policy_net.load_state_dict(torch.load("breakout-v0-beginning-test.pth"))
    target_net = DQN(em.num_actions_available(), device).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

    score_array = []
    loss_array = []

    # em.plot_screen()

    # em.reset()
    # state = em.get_state()
    # for timestep in range(20000):
    #     action = agent.select_action(state, policy_net)
    #     reward = em.take_action(action.item())
    #     next_state = em.get_state()
    #     memory.push(Experience(state, action, next_state, reward))
    #     for s in state:
    #         em.plot_screen(s)
    #     state = next_state

    # for episode in range(num_episodes):
    #     em.reset()
    #     state = em.get_state()
    #
    #     scores = []
    #     losses = []
    #     for timestep in count():
    #         action = agent.select_action(state, policy_net)
    #         reward = em.take_action(action.item())
    #         scores.append(reward)
    #         next_state = em.get_state()
    #         memory.push(Experience(state, action, next_state, reward))
    #         state = next_state
    #
    #         loss = None
    #         if memory.can_provide_sample():
    #             experiences = memory.sample()
    #             states, actions, rewards, next_states = extract_tensors(experiences)
    #
    #             current_q_values = QValues.get_current(policy_net, states, actions)
    #             next_q_values = QValues.get_next(target_net, next_states)
    #             target_q_values = (next_q_values * gamma) + rewards.to(device)
    #
    #             loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             losses.append(loss.item()/len(experiences))
    #
    #         if em.done:
    #             if memory.can_provide_sample() and episode % target_update == 0:
    #                 scores = np.asarray(scores).sum()
    #                 losses = np.asarray(losses).mean()
    #                 score_array.append(scores)
    #                 loss_array.append(losses)
    #                 print("Episode", episode)
    #                 plot_2(score_array, loss_array)
    #                 print(" Epsilon:", strategy.get_exploration_rate(agent.current_step))
    #                 print(" Memory len:", memory.push_count)
    #             break
    #
    #     if episode % target_update == 0:
    #         target_net.load_state_dict(policy_net.state_dict())
    #     torch.save(policy_net.state_dict(), 'sdaw')

    policy_net.eval()
    mean_score = 0
    for episode in range(20):
        em.reset()
        state = em.get_state()

        scores = []
        for t in range(150):
            em.render()
            action = agent.select_action(state, policy_net, optimal=True)
            reward = em.take_action(action.item())
            scores.append(reward)
            state = em.get_state()
            # time.sleep(.05)
            if em.done:
                break
        print("Episode {} finished after {} timesteps with {} scores".format(episode, t + 1, np.sum(scores)))
        mean_score = (mean_score * episode + np.sum(scores)) / (episode + 1)

    print("Mean score", mean_score)
    em.close()
