import sys
import time
import numpy as np

sys.path.append("./highway-env/")

import highway_env
import gymnasium as gym
from stable_baselines3 import PPO

# Parallel environments
env = gym.make("merge-single-agent-v0")

model = PPO.load("ppo_car_following_straight.zip")
# model = PPO.load("ppo_car_following_circular.zip")

num_runs = 10000

ttcs = []
headways = []

obs, _ = env.reset()
for _ in range(num_runs):
    action, _ = model.predict(obs)
    obs, reward, done, _, info = env.step(action)
    # print(info)
    ttcs.append(info["ttc"])
    headways.append(info["headway"])
    if done:
        obs, _ = env.reset()
    env.render()
    time.sleep(1 / env.unwrapped.config["policy_frequency"])

# np.save("ttc_straight.npy", np.array(ttcs))
# np.save("headway_straight.npy", np.array(headways))
