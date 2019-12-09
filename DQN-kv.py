from itertools import count

import torch.optim as optim

from DefineClasses import *
from utils import *

if __name__ == "__main__":
    environment = 'Breakout-v0'
    batch_size = 2048
    gamma = 0.95
    eps_start = 1
    eps_end = 0.1
    eps_decay = 0.0001
    target_update = 1
    memory_size = 100000
    lr = 0.001
    num_episodes = 500
    k = 4  # of skipped frames

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    em = EnvManager(environment, device, k=k)
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
    memory = ReplayMemory(memory_size, batch_size)
    agent = Agent(strategy, em.num_actions_available(), device, memory)

    policy_net = DQN(em.num_actions_available(), device).to(device)
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
    #     reward = em.take_action(action)
    #     next_state = em.get_state()
    #     memory.push(Experience(state, action, next_state, reward))
    #     for s in state:
    #         em.plot_screen(s)
    #     state = next_state

    # policy_net.load_state_dict(torch.load("breakout-v0-1d.pth"))
    for episode in range(num_episodes):
        em.reset()
        state = em.get_state()

        scores = []
        losses = []
        for timestep in count():
            action = agent.select_action(state, policy_net)
            reward = em.take_action(action)
            scores.append(reward)
            next_state = em.get_state()
            memory.push(Experience(state, action, next_state, reward))
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
                losses.append(loss.item() * 1000)

            if em.done:
                scores = np.asarray(scores).sum()
                losses = np.asarray(losses).mean()
                score_array.append(scores)
                loss_array.append(losses)
                plot_2(score_array, loss_array)
                print(" Epsilon:", strategy.get_exploration_rate(agent.current_step))
                print(" Memory len:", memory.push_count)
                break

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        torch.save(policy_net.state_dict(), 'breakout-v0-1e.pth')

    policy_net.eval()
    for episode in range(20):
        em.reset()
        state = em.get_state()

        scores = []
        for t in range(300):
            em.render()
            action = agent.select_action(state, policy_net, optimal=True)
            reward = em.take_action(action)
            scores.append(reward)
            state = em.get_state()
            if em.done:
                print("Episode finished after {} timesteps with {} scores".format(t + 1, np.sum(scores)))
                break
    em.close()
