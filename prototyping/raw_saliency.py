import time
import torch
import torch.nn as nn

# from pytorch_grad_cam.utils.model_targets import BaseCAMTarget
import numpy as np
import cv2
import gymnasium as gym
from stable_baselines3 import PPO

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


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained agent
# model = PPO.load("logs/01-02:11-16:48.89/best_model/best_model.zip", device=device)
model = PPO.load("./logs/01-09:15-11:21.25/best_model/best_model.zip", device=device)

# Wrap policy
wrapped_policy = PPOContinuousPolicyWrapper(model.policy).to(device)
wrapped_policy.eval()


env = gym.make("merge-single-agent-v0")
# Get observation
obs, _ = env.reset()  # shape (1, 64, 64)
obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
obs_tensor.requires_grad_(True)

for i in range(5000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)

    if done:
        obs, _ = env.reset()

    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    obs_tensor.requires_grad_(True)

    action_mean = wrapped_policy(obs_tensor)  # [1, 1]
    speed = action_mean[0, 0]

    # Clear old gradients
    wrapped_policy.zero_grad()

    # Backpropagate
    speed.backward()

    saliency = obs_tensor.grad.detach().abs()
    saliency = saliency.squeeze().cpu().numpy()  # [64, 64]

    # Normalize for visualization
    saliency /= saliency.max() + 1e-8

    # Normalize radar image for overlay
    radar = obs[0]  # (64, 64)
    radar_norm = (radar - radar.min()) / (radar.max() - radar.min() + 1e-6)
    radar_rgb = np.repeat(radar_norm[..., None], 3, axis=2)

    saliency_color = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_JET)

    overlay = 0.6 * radar_rgb + 0.4 * (saliency_color / 255.0)
    overlay = np.clip(overlay, 0, 1)

    # cv2.imwrite("radar_gradcam_speed.png", cam_image[:, :, ::-1])
    # cv2.imwrite("radar_base.png", radar)
    cv2.imshow(
        "grdcam",
        cv2.flip(
            cv2.resize((overlay * 255).astype(np.uint8)[:, :, ::-1], (900, 900)), 0
        ),
    )
    # cv2.imshow("radar base", cv2.flip(cv2.resize(radar, (900, 900)), 0))

    cv2.waitKey(1)
    env.render()
    time.sleep(0.1)
