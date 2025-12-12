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


def now_str():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    return now.strftime("%m-%d:%H-%M:%S.%f")[:-4]


def init_seeding(seed):
    np.random.seed(seed)


def init_log_folder():
    log_folder = os.path.join("logs", now_str())
    os.makedirs(log_folder, exist_ok=True)
    os.system(f"cp -r ./highway-env/ {log_folder}")

    return log_folder


def init_logging(log_folder):
    run = wandb.init(
        project="car-following-env",
        name=log_folder,
        # config=config,
        sync_tensorboard=True,
        save_code=True,
    )

    return run


def init_callbacks(env, log_folder):
    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"{log_folder}/best_model",
        log_path="{log_folder}/results",
        eval_freq=10000,
    )

    wandb_callback = WandbCallback(
        # gradient_save_freq=100,
        # model_save_path=f"models/{run.id}",
        verbose=2
    )

    callback = CallbackList([eval_callback, wandb_callback])

    return callback


def init_agent(env, log_folder):
    policy_kwargs = dict(
        features_extractor_kwargs=dict(features_dim=512),
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
        tensorboard_log=f"{log_folder}",
        seed=seed,
    )

    return model


def main():
    init_seeding(seed)
    log_folder = init_log_folder()
    wandb_run = init_logging(log_folder)

    env = gym.make("merge-single-agent-v0")

    model = init_agent(env, log_folder)
    callback = init_callbacks(env, log_folder)

    model.learn(total_timesteps=1e6, callback=callback)
    model.save(f"{log_folder}/model.zip")

    wandb_run.finish()


if __name__ == "__main__":
    main()
