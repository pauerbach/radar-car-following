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
seed = 72

np.random.seed(seed)

config = {
    "action": {"type": "IDM"},
    "observation": {"type": "Kinematics"},
}
env = gym.make("merge-single-agent-v0", config=config)

num_runs = 50000

ttcs = []
headways = []
leader_speeds = []
dists = []
crashes = 0
episodes = 0

# max_range = env.unwrapped.observation_type.radar_simulator.get_max_range()
# max_doppler = env.unwrapped.observation_type.radar_simulator.get_max_velocity()
# draw = Draw(max_doppler, max_range)

t = tqdm(range(num_runs))
obs, _ = env.reset()

# model = PPO.load("./logs/01-09:15-11:21.25/best_model/best_model.zip")


for i in t:
    # action, _ = model.predict(obs, deterministic=True)
    action = 0
    obs, reward, done, _, info = env.step(action)
    # draw.draw(obs[0, :, :])
    # leader_speeds.append(info["leader_speed"])

    ttcs.append(info["ttc"])
    headways.append(info["headway"])
    if done:
        episodes += 1
        if info["crashed"]:
            crashes += 1
        obs, _ = env.reset()
    # env.render()
    # time.sleep(0.16)
    t.set_description(f"Crashes {crashes} Episodes: {episodes}")

print(f"Crashes: {crashes}")

# np.save("results/leader_speeds.npy", np.array(leader_speeds))
np.save("results/ttc_idm_with_action_noise_085_banana.npy", np.array(ttcs))
np.save("results/headway_idm_with_action_noise_085_banana.npy", np.array(headways))
