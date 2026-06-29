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

config = {}
#config = {
    #"action": {"type": "IDM"},
    #"observation": {"type": "Kinematics"},
#}
config["leader_speed_file"] = "results/leader_speeds_for_playback.npy"

env = gym.make("merge-single-agent-v0", config=config)

num_runs = 50000

ttcs = []
headways = []
leader_speeds = []
ego_speeds = []
dists = []
crashes = 0
episodes = 0

# max_range = env.unwrapped.observation_type.radar_simulator.get_max_range()
# max_doppler = env.unwrapped.observation_type.radar_simulator.get_max_velocity()
# draw = Draw(max_doppler, max_range)

t = tqdm(range(num_runs))
obs, _ = env.reset()

model = PPO.load("./logs/01-09:15-11:21.25/best_model/best_model.zip")  # RD Map input
# model = PPO.load("./logs/02-02:18-24:45.37/best_model/best_model.zip")  # direct input
# model = PPO.load("./logs/02-02:18-24:45.37/best_model/best_model.zip")
# model = PPO.load("./logs/02-03:10-34:39.76/best_model/best_model.zip")
# model = PPO.load("./logs/02-03:16-04:11.10/best_model/best_model.zip")
# model = PPO.load("./logs/02-17:14-05:37.57/best_model/best_model.zip")
# model = PPO.load("./logs/02-26:16-57:07.77/best_model/best_model.zip")  # CFAR input

# model = PPO.load("./logs/05-08:21-09:50.35/best_model/best_model.zip")  # Slot Attention

# print(model.policy)
# exit(0)

step = 0

for i in t:
    action, _ = model.predict(obs, deterministic=True)
    #action = 0
    obs, reward, done, _, info = env.step(action)
    step += 1
    # draw.draw(obs[0, :, :])
    leader_speeds.append(info["leader_speed"])
    ego_speeds.append(info["ego_speed"])

    ttcs.append(info["ttc"])
    headways.append(info["headway"])

    # if info["ttc"] < 0.5:
    #     print(f"Low TTC {info['ttc']}")
    #     print(f"Step {step}")
    # ego = env.unwrapped.road.vehicles[0]
    # leader = env.unwrapped.road.vehicles[1]
    # d = env.unwrapped._compute_headway_distance(ego)
    # print(f"Ego {ego.position}")
    # print(f"Leader {leader.position}")
    # print(f"Distance {d}")
    # dist = ego.lane.distance_between_points(ego.position, leader.position)
    # print(f"Distance {dist}")
    # ego_v = ego.speed
    # leader_v = leader.speed
    # print(f"{ego_v} {leader_v}")
    # env.render()
    # time.sleep(2)

    if done:
        episodes += 1
        if info["crashed"]:
            crashes += 1
        obs, _ = env.reset()
        step = 0
    # env.render()
    # time.sleep(0.016)

    t.set_description(f"Crashes {crashes} Episodes: {episodes}")

print(f"Crashes: {crashes}")

np.save("results/ttc_radar_for_trajectory_plot_seed42.npy", np.array(ttcs))
np.save("results/headway_radar_for_trajectory_plot_seed42.npy", np.array(headways))

np.save("results/leader_speeds_radar_for_trajectory_plot_seed42.npy", np.array(leader_speeds))
np.save("results/ego_speeds_radar_for_trajectory_plot_seed42.npy", np.array(ego_speeds))
