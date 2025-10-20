import sys

sys.path.append("./highway-env/")

import highway_env
import gymnasium as gym
import time

env = gym.make("merge-single-agent-v0")

env.reset()
start = time.time()
for _ in range(1000):
    # action = env.action_space.sample()
    action = 0.0
    obs, reward, done, _, info = env.step(action)
    # print(obs)
    if done:
        env.reset()
    env.render()
    time.sleep(1 / env.config["policy_frequency"])
print(f"Took {(time.time()-start)/1000} s")
