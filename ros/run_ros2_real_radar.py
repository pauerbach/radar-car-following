import sys
from functools import partial

import numpy as np

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Pose2D
from radar_msgs.msg import RadarFrame
from rclpy.qos import qos_profile_sensor_data


sys.path.append("./highway-env/")

from highway_env.envs.common.observation import RadarPipeline, Draw

import highway_env
import gymnasium as gym
from stable_baselines3 import PPO


class CarPublisher(Node):
    def __init__(self, use_real_radar=False, use_idm=False):
        super().__init__("car_publisher")

        # vehicle ids of the physical cars and the ids in the simulation
        self.physical_ego_id = 6
        self.sim_ego_id = 0
        self.physical_leader_id = 2
        self.sim_leader_id = 1

        self.use_real_radar = use_real_radar

        self.last_stamp = None

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

        if use_idm:
            env_config["action"]["type"] = "IDM"

        env_config["use_ros"] = True

        self.radar_pipeline = None
        self.draw = None
        self.draw_sim = None
        self.mti_history = None
        self.mti_alpha = 1.0

        self.radar_list = list()

        self.env = gym.make("merge-single-agent-v0", config=env_config)
        self.obs, _ = self.env.reset()
        self.real_obs = None

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
            qos_profile=qos_profile_sensor_data,
        )

        # self.model = PPO.load("./logs/01-02:11-16:48.89/best_model/best_model.zip")
        self.model = PPO.load("./logs/01-09:15-11:21.25/best_model/best_model.zip")
        # self.model = PPO.load("./logs/02-17:14-05:37.57/best_model/best_model.zip")

        self.sim_timer = self.create_timer(
            1 / self.env.unwrapped.config["policy_frequency"], self.timer_callback
        )

    def radar_callback(self, msg):
        self.radar_list.append(msg)

        self.last_stamp = msg.header.stamp

        if self.radar_pipeline == None:
            self.radar_pipeline = RadarPipeline(
                msg.num_antenna, msg.num_chirps, msg.num_samples
            )
            max_range = msg.max_range
            max_doppler = msg.max_doppler
            # self.draw = Draw(max_doppler, max_range, "Real Radar")
            # self.draw_sim = Draw(max_doppler, max_range, "Sim Radar")
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

        self.real_obs = obs

        # self.draw.draw(obs[0, :, :])
        # self.draw_sim.draw(self.obs[0, :, :])

        if self.use_real_radar and self.real_obs is not None:
            action, _ = self.model.predict(self.real_obs)
        else:
            action, _ = self.model.predict(self.obs)

        # self.obs, _, done, _, _ = self.env.step(action)
        self.env.unwrapped._simulate(action)

        # self.env.render()

    def timer_callback(self):
        pass
        # if self.use_real_radar and self.real_obs is not None:
        #     action, _ = self.model.predict(self.real_obs)
        # else:
        #     action, _ = self.model.predict(self.obs)
        #
        # # self.obs, _, done, _, _ = self.env.step(action)
        # self.env.unwrapped._simulate(action)
        #
        # self.env.render()

    def pose_callback(self, msg, id):
        position = np.zeros(2)
        position[0] = msg.x
        position[1] = msg.y

        heading = msg.theta
        heading = highway_env.utils.wrap_to_pi(heading)
        self.env.unwrapped.road.vehicles[id].set_pose(position, heading)

    def publish(self, steering, speed, id):
        msg = AckermannDriveStamped()

        t = self.get_clock().now()
        msg.header.stamp = t.to_msg()
        # msg.header.stamp = self.last_stamp
        msg.drive.speed = speed
        msg.drive.steering_angle = np.rad2deg(steering)
        self.pubs[id].publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = CarPublisher(use_real_radar=True, use_idm=False)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Interrupted")


if __name__ == "__main__":
    main()
