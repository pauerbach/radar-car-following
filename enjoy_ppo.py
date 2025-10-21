import sys
import time

sys.path.append("./highway-env/")

import highway_env
import gymnasium as gym
from stable_baselines3 import PPO

# Parallel environments
env = gym.make("merge-single-agent-v0")

model = PPO.load("ppo_car_following_straight.zip")

num_runs = 100000

obs, _ = env.reset()
for _ in range(num_runs):
    action, _ = model.predict(obs)
    obs, reward, done, _, info = env.step(action)
    if done:
        obs, _ = env.reset()
    env.render()
    time.sleep(1 / env.unwrapped.config["policy_frequency"])
