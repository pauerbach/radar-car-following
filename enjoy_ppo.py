import sys
import time
import numpy as np

sys.path.append("./highway-env/")

import highway_env
import gymnasium as gym
from stable_baselines3 import PPO

# Parallel environments
env = gym.make("merge-single-agent-v0")

# model = PPO.load("results/ppo_car_following_straight.zip")
# model = PPO.load("results/ppo_car_following_circular.zip")
# model = PPO.load("results/ppo_car_following_waving.zip")
# model = PPO.load("results/ppo_car_following_straight_no_normalization_10hz.zip")
model = PPO.load(
    "./results/ppo_car_following_circular_no_normalization_10hz_vel_discretization_added_noise.zip"
)

num_runs = 10000

ttcs = []
headways = []
leader_speeds = []

obs, _ = env.reset()
for _ in range(num_runs):
    action, _ = model.predict(obs)
    obs, reward, done, _, info = env.step(action)
    leader_speeds.append(obs[2])

    ttcs.append(info["ttc"])
    headways.append(info["headway"])
    if done:
        obs, _ = env.reset()
    env.render()
    time.sleep(1 / env.unwrapped.config["policy_frequency"])

    # env.unwrapped.road.vehicles[0].set_pose(np.array([0.0, 1.0]), 0.0)

# np.save(
#     "results/speeds_circular_10hz_discretization_noise_constant_leader.npy",
#     np.array(leader_speeds),
# )
# np.save("results/ttc_circular_10hz_discretization_noise.npy", np.array(ttcs))
# np.save("results/headway_circular_10hz_discretization_noise.npy", np.array(headways))
