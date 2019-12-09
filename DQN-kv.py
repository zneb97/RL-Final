from itertools import count

import torch.optim as optim

from DefineClasses import *
from utils import *

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

    policy_net = DQN(em.num_actions_available(), device).to(device)
    target_net = DQN(em.num_actions_available(), device).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

    episode_durations = []

    em.plot_screen()
    policy_net.load_state_dict(torch.load("breakout-v0-1c.pth"))
    for episode in range(num_episodes):
        em.reset()
        state = em.get_state()

        scores = []
        for timestep in count():

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
                action = agent.select_action(state, policy_net, optimal=True)
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
            action = agent.select_action(state, policy_net, optimal=True)
            reward = em.take_action(action)
            state = em.get_state()
            if em.done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    em.close()
