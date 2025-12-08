import sys

sys.path.append("./highway-env/")

import highway_env
import gymnasium as gym
from stable_baselines3 import PPO

# Parallel environments
env = gym.make("merge-single-agent-v0")

print("observation_space")
print(env.observation_space)
print()

model = PPO(
    # "MlpPolicy",
    "CnnPolicy",
    env,
    verbose=1,
    # n_steps=int(1e6),
    learning_rate=5e-4,
    gamma=0.99,
    vf_coef=0.5,
    clip_range=0.2,
    gae_lambda=0.95,
    n_epochs=10,
    batch_size=64,
)
model.learn(total_timesteps=1e6)
model.save("ppo_car_following")
