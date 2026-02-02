import sys
from functools import partial

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
# import seaborn as sns

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Pose2D
from radar_msgs.msg import RadarFrame


sys.path.append("./highway-env/")

from highway_env.envs.common.observation import RadarPipeline, Draw

import highway_env
import gymnasium as gym
from stable_baselines3 import PPO

import torch
import torch.nn as nn

import cv2


class SavingDraw:
    def __init__(self, max_speed_m_s, max_range_m):
        # sns.set_theme()
        # sns.set_context("paper")
        matplotlib.rcParams.update({"font.size": 18})
        self._max_speed_m_s = max_speed_m_s
        self._max_range_m = max_range_m

    def save(self, filename, data):
        print(f"Saving as {filename}")
        minmin = np.min(data)
        maxmax = np.max(data)

        plt.imshow(
            data,
            vmin=minmin,
            vmax=maxmax,
            extent=(
                -self._max_speed_m_s,
                self._max_speed_m_s,
                0,
                self._max_range_m / 2,
            ),
            # aspect="equal",
            aspect=4,
            origin="lower",
        )

        plt.xlabel("Velocity [m/s]")
        plt.ylabel("Distance [m]")
        plt.savefig(filename, dpi=600, bbox_inches="tight")
        # plt.show()


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


class CarPublisher(Node):
    def __init__(self):
        super().__init__("car_publisher")

        # vehicle ids of the physical cars and the ids in the simulation
        self.physical_ego_id = 6
        self.sim_ego_id = 0
        self.physical_leader_id = 2
        self.sim_leader_id = 1

        self.i = 0

        self.pubs = dict()

        self.pubs[self.physical_ego_id] = self.create_publisher(
            AckermannDriveStamped, f"/car{self.physical_ego_id}/cmd_vel", 10
        )
        self.pubs[self.physical_leader_id] = self.create_publisher(
            AckermannDriveStamped, f"/car{self.physical_leader_id}/cmd_vel", 10
        )

        env_config = highway_env.envs.SingleAgentMergeEnv.default_config()

        env_config["ego_callback"] = partial(self.publish, id=self.physical_ego_id)
        env_config["leader_callback"] = partial(
            self.publish, id=self.physical_leader_id
        )

        env_config["action"]["add_noise"] = False
        env_config["use_ros"] = True

        self.radar_pipeline = None
        self.draw = None
        self.draw_sim = None
        self.mti_history = None
        self.mti_alpha = 1.0

        self.radar_list = list()

        self.env = gym.make("merge-single-agent-v0", config=env_config)
        self.obs, _ = self.env.reset()

        # ego pose subscription
        self.create_subscription(
            Pose2D,
            f"/car{self.physical_ego_id}/ground_pose",
            partial(self.pose_callback, id=self.sim_ego_id),
            10,
        )
        # leader pose subscription
        self.create_subscription(
            Pose2D,
            f"/car{self.physical_leader_id}/ground_pose",
            partial(self.pose_callback, id=self.sim_leader_id),
            10,
        )

        # subscription for radar data
        self.create_subscription(
            RadarFrame,
            "radar_data",
            self.radar_callback,
            10,
        )

        self.model = PPO.load("./logs/01-02:11-16:48.89/best_model/best_model.zip")

        self.wrapped_policy = PPOContinuousPolicyWrapper(self.model.policy)
        self.wrapped_policy.eval()

        self.sim_timer = self.create_timer(
            1 / self.env.unwrapped.config["policy_frequency"], self.timer_callback
        )

    def radar_callback(self, msg):
        self.radar_list.append(msg)

        if self.radar_pipeline == None:
            self.radar_pipeline = RadarPipeline(
                msg.num_antenna, msg.num_chirps, msg.num_samples
            )
            max_range = msg.max_range
            max_doppler = msg.max_doppler
            # self.draw = Draw(max_doppler, max_range)
            self.save_draw = SavingDraw(max_doppler, max_range)
            # self.draw_sim = Draw(max_doppler, max_range)
            self.save_draw_sim = SavingDraw(max_doppler, max_range)
            self.mti_history = np.zeros(
                (msg.num_samples, msg.num_chirps, msg.num_antenna)
            )

        radar_cube = np.array(msg.data).reshape(
            msg.num_antenna, msg.num_chirps, msg.num_samples
        )

        radar_cube = np.swapaxes(radar_cube, 0, 2)
        radar_cube -= np.mean(radar_cube, axis=1, keepdims=True)

        radar_cube_mti = radar_cube
        # radar_cube_mti = radar_cube - self.mti_history
        # self.mti_history = radar_cube * self.mti_alpha + self.mti_history * (
        #     1 - self.mti_alpha
        # )

        doppler_fft = self.radar_pipeline.run(radar_cube_mti)
        obs = 20 * np.log10(np.abs(np.average(doppler_fft, 2)) + 1e-12)
        obs = obs[:-1, :]

        # TODO add normilzation based on overall time min and max for more consistent output
        # normalize
        minmin = np.min(obs)
        maxmax = np.max(obs)
        obs = (obs - minmin) * 255 / (maxmax - minmin)
        obs = obs[np.newaxis, ...].astype(np.uint8)

        self.i += 1
        # self.draw.draw(obs[0, :, :])
        self.save_draw.save(f"real_{self.i}.png", obs[0, :, :])

        # self.draw_sim.draw(self.obs[0, :, :])
        self.save_draw_sim.save(f"sim_{self.i}.png", self.obs[0, :, :])

        # Calculate saliency map
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        obs_tensor.requires_grad_(True)

        action_mean = self.wrapped_policy(obs_tensor)  # [1, 1]
        speed = action_mean[0, 0]

        # Clear old gradients
        self.wrapped_policy.zero_grad()

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

        cv2.imshow(
            "saliency",
            cv2.flip(
                cv2.resize((overlay * 255).astype(np.uint8)[:, :, ::-1], (900, 900)), 0
            ),
        )

    def timer_callback(self):
        action, _ = self.model.predict(self.obs)
        self.obs, _, done, _, _ = self.env.step(action)
        done = False

        if done:
            self.obs, _ = self.env.reset()

        self.env.render()

    def pose_callback(self, msg, id):
        position = np.zeros(2)
        position[0] = msg.x
        position[1] = msg.y

        heading = msg.theta
        heading = highway_env.utils.wrap_to_pi(heading)
        self.env.unwrapped.road.vehicles[id].set_pose(position, heading)

    def publish(self, steering, speed, id):
        msg = AckermannDriveStamped()

        msg.drive.speed = speed
        msg.drive.steering_angle = np.rad2deg(steering)
        self.pubs[id].publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = CarPublisher()
    # rclpy.spin(node)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Interrupted")
        # np.save("simulated_obs2.npy", node.obs_list)
        # np.save("real_obs.npy", node.radar_list)


if __name__ == "__main__":
    main()
