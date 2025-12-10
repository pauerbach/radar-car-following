import sys

sys.path.append("./highway-env/")

import highway_env
from highway_env.envs.common.observation import Draw
import gymnasium as gym
import time

env = gym.make("merge-single-agent-v0")

env.reset()
start = time.time()
num_runs = 10000
max_range = env.unwrapped.observation_type.radar_simulator.get_max_range()
max_doppler = env.unwrapped.observation_type.radar_simulator.get_max_velocity()
draw = Draw(max_doppler, max_range)

for _ in range(num_runs):
    # action = env.action_space.sample()
    action = 0.0
    obs, reward, done, _, info = env.step(action)
    #draw.draw(obs)

    if done:
        env.reset()

    #env.render()
    #time.sleep(1 / env.unwrapped.config["policy_frequency"])

print(f"Took {(time.time()-start)/num_runs} s")
