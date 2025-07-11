import gym
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

environment = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
observation, _ = environment.reset()

nb_states = environment.observation_space.n
nb_actions = environment.action_space.n
qtable = np.zeros((nb_states, nb_actions))

print("Q-table: ")
print(qtable)

episodes = 1000
alpha = 0.5
gamma = 0.9

outcomes = []

for i in tqdm(range(episodes)):

    state, _ = environment.reset()
    done = False
    outcomes.append("Failure")

    while not done:

        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])
        else:
            action = environment.action_space.sample()

        new_state, reward, terminated, truncated, info = environment.step(action)

        qtable[state, action] = qtable[state, action] + alpha * (
            reward + gamma * np.max(qtable[new_state]) - qtable[state, action]
        )

        state = new_state
        done = terminated or truncated

        if reward:
            outcomes[-1] = "Success"

print("Qtable After Training: ")
print(qtable)

successes = outcomes.count("Success")
failures = outcomes.count("Failure")

plt.bar(["Success", "Failure"], [successes, failures])
plt.title("Outcome Distribution over Episodes")
plt.show()
