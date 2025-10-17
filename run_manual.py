import sys

sys.path.append("./highway-env/")

import highway_env
import gymnasium as gym
import time

env = gym.make("merge-single-agent-v0")

env.reset()
for _ in range(1000):
    # action = env.action_space.sample()
    action = 0.0
    # start = time.time()
    obs, reward, done, _, info = env.step(action)
    # print(obs)
    # print(f"Took {time.time()-start} s")
    if done:
        env.reset()
    env.render()
    time.sleep(0.1)
