import time
import torch
import torch.nn as nn

# from pytorch_grad_cam.utils.model_targets import BaseCAMTarget
import numpy as np
import cv2
import gymnasium as gym
from stable_baselines3 import PPO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import sys

sys.path.append("./highway-env/")

import highway_env


class PPOContinuousPolicyWrapper(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        # Feature extraction
        features = self.policy.extract_features(obs)

        # Policy latent
        latent_pi, _ = self.policy.mlp_extractor(features)

        # Mean action (before tanh / clipping)
        action_mean = self.policy.action_net(latent_pi)
        return action_mean


class ContinuousActionTarget:
    def __init__(self, action_idx=0):
        self.action_idx = action_idx

    def __call__(self, model_output):
        # Sum keeps gradient scale stable
        # print(model_output)
        # print()
        # return model_output[:, self.action_idx].sum()
        return model_output[0]


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained agent
model = PPO.load("logs/01-02:11-16:48.89/best_model/best_model.zip", device=device)

# Wrap policy
wrapped_policy = PPOContinuousPolicyWrapper(model.policy).to(device)
wrapped_policy.eval()

# Target convolutional layer
target_layers = [model.policy.features_extractor.cnn[9]]

cam = GradCAM(
    model=wrapped_policy,
    target_layers=target_layers,
)

env = gym.make("merge-single-agent-v0")
# Get observation
obs, _ = env.reset()  # shape (1, 64, 64)
obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

for i in range(5000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)

    if done:
        obs, _ = env.reset()

    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

    # Grad-CAM target
    targets = [ContinuousActionTarget(action_idx=0)]

    # Generate saliency
    grayscale_cam = cam(input_tensor=obs_tensor, targets=targets)[
        0
    ]  # shape (8, 8), automatically upsampled

    # Normalize radar image for overlay
    radar = obs[0]  # (64, 64)
    radar_norm = (radar - radar.min()) / (radar.max() - radar.min() + 1e-6)
    radar_rgb = np.repeat(radar_norm[..., None], 3, axis=2)

    # Overlay CAM
    cam_image = show_cam_on_image(
        radar_rgb, grayscale_cam, use_rgb=True, image_weight=0.5
    )

    # cv2.imwrite("radar_gradcam_speed.png", cam_image[:, :, ::-1])
    # cv2.imwrite("radar_base.png", radar)
    cv2.imshow("grdcam", cv2.flip(cv2.resize(cam_image[:, :, ::-1], (900, 900)), 0))
    # cv2.imshow("radar base", cv2.flip(cv2.resize(radar, (900, 900)), 0))

    cv2.waitKey(1)
    env.render()
    time.sleep(0.1)
