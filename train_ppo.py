import sys
import os
import numpy as np
import datetime
import dateutil.tz

sys.path.append("./highway-env/")

import highway_env
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
import wandb
from wandb.integration.sb3 import WandbCallback

seed = 42

np.random.seed(seed)
def now_str():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    return now.strftime("%m-%d:%H-%M:%S.%f")[:-4]

env = gym.make("merge-single-agent-v0")

policy_kwargs = dict(
    features_extractor_kwargs=dict(features_dim=512),
)


log_folder = os.path.join(f"logs", now_str())
os.makedirs(log_folder, exist_ok=True)
os.system(f"cp -r ./highway-env/ {log_folder}")

run = wandb.init(
    project="car-following-env",
    name=log_folder,
    #config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

eval_callback = EvalCallback(env, best_model_save_path=f"{log_folder}/best_model",
                                     log_path="{log_folder}/results", eval_freq=10000)

wandb_callback=WandbCallback(
    #gradient_save_freq=100,
    #model_save_path=f"models/{run.id}",
    verbose=2
)

callback = CallbackList([eval_callback, wandb_callback])

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
    tensorboard_log=f"{log_folder}",
    seed=seed,
)

model.learn(total_timesteps=1e6,
            callback=callback
            )
model.save(f"{log_folder}/model.zip")

run.finish()
