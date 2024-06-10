import numpy as np
from collections import defaultdict


class Solver():
    def __init__(self, env):
        self.env = env
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    def train(self, num_episodes=10000, max_steps_per_episode=100,  beta=0.1, gamma=0.8):
        rewards_per_episode = []
        steps_per_episode = []

        for episode in range(num_episodes):
            state = self.env.reset()[0]
            done = False
            steps = 0
            total_reward = 0

            while not done and steps < max_steps_per_episode:
                action = np.argmax(self.Q[state])
                next_state, reward, done, _, _ = self.env.step(action)

                best_next_action = np.argmax(self.Q[next_state])
                td_error = reward + gamma * self.Q[next_state][best_next_action] - self.Q[state][action]
                self.Q[state][action] += beta * td_error

                state = next_state
                total_reward += reward
                steps += 1

            rewards_per_episode.append(total_reward)
            steps_per_episode.append(steps)
        
        return rewards_per_episode, steps_per_episode
