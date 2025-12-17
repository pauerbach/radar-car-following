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

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
import omegaconf

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class RadarCNN64x64(BaseFeaturesExtractor):
    """
    Optimized CNN feature extractor for a 64x64 range–Doppler map
    with a single magnitude channel.


    Observation space:
    Box(shape=(1, 64, 64))


    Design principles:
    - Exploit fixed spatial resolution (64x64)
    - Aggressive early downsampling for stability
    - Conservative capacity (single-target car following)
    - LayerNorm instead of BatchNorm (on-policy PPO)
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        assert observation_space.shape == (1, 64, 64), (
            "RadarCNN64x64 expects observation shape (1, 64, 64), "
            f"got {observation_space.shape}"
        )

        # -----------------------------
        # Convolutional backbone
        # -----------------------------
        self.cnn = nn.Sequential(
            # (1, 64, 64) -> (16, 32, 32)
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.LayerNorm([16, 32, 32]),
            nn.ReLU(),
            # (16, 32, 32) -> (32, 16, 16)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([32, 16, 16]),
            nn.ReLU(),
            # (32, 16, 16) -> (64, 8, 8)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([64, 8, 8]),
            nn.ReLU(),
            # Preserve spatial structure but reduce sensitivity
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([64, 8, 8]),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Output of CNN is 64 * 8 * 8 = 4096
        self.linear = nn.Sequential(
            nn.Linear(4096, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def now_str():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    return now.strftime("%m-%d:%H-%M:%S.%f")[:-4]


def init_seeding(seed):
    np.random.seed(seed)


def init_log_folder():
    log_folder = os.path.join(f"{to_absolute_path('logs')}", now_str())
    os.makedirs(log_folder, exist_ok=True)
    os.system(f"cp -r {to_absolute_path('highway-env')}/ {log_folder}")
    os.system(f"cp  {to_absolute_path('config.yaml')} {log_folder}")

    return log_folder


def init_logging(log_folder, cfg):
    run = wandb.init(
        project="car-following-env",
        name=log_folder,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        sync_tensorboard=True,
        save_code=True,
    )
    artifact = wandb.run.log_code(
        f"{to_absolute_path('highway-env')}",
        name="Simulation_Code",
        include_fn=lambda path: path.endswith(".py") or path.endswith(".pyx"),
    )
    artifact.wait()
    wandb.run.use_artifact(artifact, type="code")

    return run


def init_callbacks(env, log_folder):
    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"{log_folder}/best_model",
        # log_path=f"{log_folder}/results",
        eval_freq=10000,
    )

    wandb_callback = WandbCallback(verbose=2)

    callback = CallbackList([eval_callback, wandb_callback])

    return callback


def init_agent(env, log_folder, cfg, seed):
    policy_kwargs = dict(
        features_extractor_class=RadarCNN64x64,
        features_extractor_kwargs=dict(features_dim=cfg.features_dim),
    )

    model = PPO(
        cfg.policy,
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        vf_coef=cfg.vf_coef,
        clip_range=cfg.clip_range,
        gae_lambda=cfg.gae_lambda,
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        tensorboard_log=f"{log_folder}",
        seed=seed,
    )

    return model


@hydra.main(version_base="1.1", config_path=".", config_name="config")
def main(cfg: "DictConfig"):
    print(cfg)
    print(os.getcwd())
    init_seeding(cfg.seed)
    log_folder = init_log_folder()
    wandb_run = init_logging(log_folder, cfg)

    env_cfg = omegaconf.OmegaConf.to_container(
        cfg.env, resolve=True, throw_on_missing=True
    )
    env = gym.make("merge-single-agent-v0", config=env_cfg)

    model = init_agent(env, log_folder, cfg.train, cfg.seed)
    callback = init_callbacks(
        env,
        log_folder,
    )

    model.learn(total_timesteps=cfg.train.total_timesteps, callback=callback)
    model.save(f"{log_folder}/model.zip")

    wandb_run.finish()


if __name__ == "__main__":
    main()
