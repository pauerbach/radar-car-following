import sys
from functools import partial

import numpy as np

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


class CarPublisher(Node):
    def __init__(self):
        super().__init__("car_publisher")

        # vehicle ids of the physical cars and the ids in the simulation
        self.physical_ego_id = 6
        self.sim_ego_id = 0
        self.physical_leader_id = 2
        self.sim_leader_id = 1

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

        self.radar_pipeline = None
        self.draw = None
        self.mti_history = None
        self.mti_alpha = 1.0

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

        self.model = PPO.load(
            "./results/ppo_car_following_circular_no_normalization_10hz_vel_discretization_added_noise.zip"
        )

        # self.sim_timer = self.create_timer(
        #     1 / self.env.unwrapped.config["policy_frequency"], self.timer_callback
        # )

    def radar_callback(self, msg):
        if self.radar_pipeline == None:
            self.radar_pipeline = RadarPipeline(
                msg.num_antenna, msg.num_chirps, msg.num_samples
            )
            max_range = msg.max_range
            max_doppler = msg.max_doppler
            self.draw = Draw(max_doppler, max_range)
            self.mti_history = np.zeros(
                (msg.num_samples, msg.num_chirps, msg.num_antenna)
            )

        radar_cube = np.array(msg.data).reshape(
            msg.num_antenna, msg.num_chirps, msg.num_samples
        )

        radar_cube = np.swapaxes(radar_cube, 0, 2)
        radar_cube -= np.mean(radar_cube, axis=0, keepdims=True)

        radar_cube_mti = radar_cube - self.mti_history
        self.mti_history = radar_cube * self.mti_alpha + self.mti_history * (
            1 - self.mti_alpha
        )

        doppler_fft = self.radar_pipeline.run(radar_cube_mti)
        obs = 20 * np.log10(np.abs(np.average(doppler_fft, 2)) + 1e-12)

        # TODO add normilzation based on overall time min and max for more consistent output
        # normalize
        minmin = np.min(obs)
        maxmax = np.max(obs)
        obs = (obs - minmin) * 255 / (maxmax - minmin)
        obs = obs[np.newaxis, ...].astype(np.uint8)

        self.draw.draw(obs[0, :, :])
        # self.draw.draw(obs)

    def timer_callback(self):
        # action, _ = self.model.predict(self.obs)
        # self.obs, _, done, _, _ = self.env.step(action)
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
    rclpy.spin(node)


if __name__ == "__main__":
    main()
