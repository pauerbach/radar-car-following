import sys
from functools import partial

import numpy as np

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Pose2D
from pcl_radar_msgs.msg import RadarTrack

sys.path.append("./highway-env/")

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

        self.obs_list = []
        self.radar_list = []

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

        self.env = gym.make("merge-single-agent-v0", config=env_config)
        self.obs, _ = self.env.reset()

        self.obs_list.append(self.obs)

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

        self.create_subscription(
            RadarTrack,
            "/radar_track",
            self.log_radar,
            10,
        )

        self.model = PPO.load(
            "./results/ppo_car_following_circular_no_normalization_10hz_vel_discretization_added_noise.zip"
        )

        self.sim_timer = self.create_timer(
            1 / self.env.unwrapped.config["policy_frequency"], self.timer_callback
        )

    def log_radar(self, msg):
        print("got radar msg")
        msg_np = np.array([msg.x, msg.y, msg.vel])
        self.radar_list.append(msg_np)

    def timer_callback(self):
        action, _ = self.model.predict(self.obs)
        self.obs, _, done, _, _ = self.env.step(action)

        self.obs_list.append(self.obs)

        if done:
            self.obs, _ = self.env.reset()
            self.obs_list.append(self.obs)

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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Interrupted")
        np.save("simulated_obs2.npy", node.obs_list)
        np.save("real_obs2.npy", node.radar_list)


if __name__ == "__main__":
    main()
