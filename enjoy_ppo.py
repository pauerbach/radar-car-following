import sys
import time
import numpy as np
from tqdm import tqdm

sys.path.append("./highway-env/")

import highway_env
from highway_env.envs.common.observation import Draw
import gymnasium as gym
from stable_baselines3 import PPO


## Set same seed for reproducability
seed = 42

np.random.seed(seed)

# Parallel environments
env = gym.make("merge-single-agent-v0")

model = PPO.load(
    # "./results/ppo_car_following_circular_no_normalization_10hz_vel_discretization_added_noise.zip"
    # "./results/ppo_car_following_circular_no_normalization_10hz_vel_discretization_added_noise_distance_only.zip"
    # "./results/ppo_car_following_waving_no_normalization_10hz_vel_discretization_added_noise_distance_only.zip"
    # "./results/ppo_car_following_radar_input.zip"
    # "./results/ppo_car_following_radar_1_8_termination.zip"
    # "./results/ppo_car_following_radar_1_8_termination_no_zero_noise.zip"
    # "./results/ppo_car_following_dist_only_no_ego_speed.zip"
    # "./results/ppo_car_following_dist_only_with_ego_speed.zip"
    # "./results/ppo_car_following_radar_2_feature_dim.zip"
    # "./results/ppo_car_following_radar_64_feature_dim.zip"
    "./results/ppo_car_following_radar_128_feature_dim.zip"
)

num_runs = 50000
# num_runs = 5000

ttcs = []
headways = []
leader_speeds = []
dists = []
crashes = 0

# max_range = env.unwrapped.observation_type.radar_simulator.get_max_range()
# max_doppler = env.unwrapped.observation_type.radar_simulator.get_max_velocity()
# draw = Draw(max_doppler, max_range)

t = tqdm(range(num_runs))
obs, _ = env.reset()
for _ in t:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    # draw.draw(obs[0, :, :])
    # leader_speeds.append(obs[2])
    # dists.append(obs[0])

    ttcs.append(info["ttc"])
    headways.append(info["headway"])
    if done:
        if info["crashed"]:
            crashes += 1
        obs, _ = env.reset()
    # env.render()
    # time.sleep(1 / env.unwrapped.config["policy_frequency"])
    # time.sleep(0.016)
    t.set_description(f"Crashes {crashes}")

    # env.unwrapped.road.vehicles[0].set_pose(np.array([0.0, 1.0]), 0.0)
print(f"Crashes: {crashes}")

# np.save(
#     "results/speeds_circular_10hz_discretization_noise_constant_leader.npy",
#     np.array(leader_speeds),
# )
# np.save("results/ttc_radar_64_feature_dim.npy", np.array(ttcs))
# np.save("results/headway_radar_64_feature_dim.npy", np.array(headways))
# np.save("results/dists_dist_only.npy", np.array(headways))
