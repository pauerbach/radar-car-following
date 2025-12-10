import sys
import numpy as np

sys.path.append("./highway-env/")

import highway_env
import gymnasium as gym
from stable_baselines3 import PPO
import wandb
from wandb.integration.sb3 import WandbCallback

seed = 42

np.random.seed(seed)

env = gym.make("merge-single-agent-v0")

policy_kwargs = dict(
    features_extractor_kwargs=dict(features_dim=128),
)

run = wandb.init(
    project="car-following-env",
    # config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

model = PPO(
    # "MlpPolicy",
    "CnnPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    learning_rate=5e-4,
    gamma=0.99,
    vf_coef=0.5,
    clip_range=0.2,
    gae_lambda=0.95,
    n_epochs=10,
    batch_size=64,
    tensorboard_log=f"runs/{run.id}",
    seed=seed,
)

model.learn(
    total_timesteps=1e6,
    callback=WandbCallback(
        gradient_save_freq=100, model_save_path=f"models/{run.id}", verbose=2
    ),
)
model.save("ppo_car_following_radar_32_feature_dim")

run.finish()
