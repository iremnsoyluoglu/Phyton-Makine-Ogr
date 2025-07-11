import gym
import random
import numpy as np

environment = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
observation, _ = environment.reset()

nb_states = environment.observation_space.n
nb_actions = environment.action_space.n
qtable = np.zeros((nb_states, nb_actions))

print("Q-table: ")
print(qtable)

action = environment.action_space.sample()
new_state, reward, terminated, truncated, info = environment.step(action)

done = terminated or truncated

print("\nYeni Durum:", new_state)
print("Ödül:", reward)
print("Bitti mi:", done)
print("Bilgi:", info)

output = environment.render()
print("\nÇevre görünümü:\n")
print(output)
