import sys

sys.path.append("./highway-env/")

import highway_env
import gymnasium as gym
import time

env = gym.make("merge-single-agent-v0")

env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    # action = 1.0
    start = time.time()
    env.step(action)
    end = time.time()
    # print(f"Took {end-start}s")
    env.render()
    time.sleep(0.1)
