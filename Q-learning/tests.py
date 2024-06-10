from solver import Solver
import matplotlib.pyplot as plt
import gymnasium as gym


def test_beta_impact():
    env = gym.make('Taxi-v3', render_mode='ansi')
    solver = Solver(env)
    rewards_beta_01, _ = solver.train(num_episodes=10000, beta=0.1)
    solver = Solver(env)
    rewards_beta_05, _ = solver.train(num_episodes=10000, beta=0.5)
    solver = Solver(env)
    rewards_beta_09, _ = solver.train(num_episodes=10000, beta=0.9)

    plt.plot(rewards_beta_01, label='Beta=0.1')
    plt.plot(rewards_beta_05, label='Beta=0.5')
    plt.plot(rewards_beta_09, label='Beta=0.9')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode for different Beta values')
    plt.legend()
    plt.show()


def test_gamma_impact():
    env = gym.make('Taxi-v3', render_mode='ansi')
    solver = Solver(env)
    rewards_gamma_01, _ = solver.train(num_episodes=10000, gamma=0.1)
    solver = Solver(env)
    rewards_gamma_05, _ = solver.train(num_episodes=10000, gamma=0.5)
    solver = Solver(env)
    rewards_gamma_09, _ = solver.train(num_episodes=10000, gamma=0.9)
    solver = Solver(env)
    rewards_gamma_099, _ = solver.train(num_episodes=10000, gamma=0.99)
    plt.plot(rewards_gamma_01, label='Gamma=0.1')
    plt.plot(rewards_gamma_05, label='Gamma=0.5')
    plt.plot(rewards_gamma_09, label='Gamma=0.9')
    plt.plot(rewards_gamma_099, label='Gamma=0.99')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode for different Gamma values')
    plt.legend()
    plt.show()

def test_Solver():
    env = gym.make('Taxi-v3', render_mode='ansi')
    solver = Solver(env)
    rewards, steps = solver.train(num_episodes=10000, max_steps_per_episode=100, beta=0.5, gamma=0.99)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')

    plt.subplot(1, 2, 2)
    plt.plot(steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps per Episode')
    plt.title('Steps per Episode')

    plt.show()


if __name__ == "__main__":
    test_beta_impact()
    test_gamma_impact()
    test_Solver()