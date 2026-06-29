import sys
from functools import partial
import time

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
    def __init__(self, use_real_radar=False):
        super().__init__("car_publisher")

        # vehicle ids of the physical cars and the ids in the simulation
        self.physical_ego_id = 6
        self.sim_ego_id = 0
        self.physical_leader_id = 2
        self.sim_leader_id = 1

        self.use_real_radar = use_real_radar

        self.ttcs = []
        self.headways = []

        self.last_time = [0, 0]
        self.last_pose = [np.zeros(2), np.zeros(2)]
        self.counter = [0, 0]

        self.actual_speeds = [[], []]
        self.commanded_speeds = [[], []]

        self.pubs = dict()

        # self.pubs[self.physical_ego_id] = self.create_publisher(
        #     AckermannDriveStamped, f"/car{self.physical_ego_id}/cmd_vel", 10
        # )
        # self.pubs[self.physical_leader_id] = self.create_publisher(
        #     AckermannDriveStamped, f"/car{self.physical_leader_id}/cmd_vel", 10
        # )

        env_config = highway_env.envs.SingleAgentMergeEnv.default_config()

        # env_config["ego_callback"] = partial(self.publish, id=self.physical_ego_id)
        # env_config["leader_callback"] = partial(
        #     self.publish, id=self.physical_leader_id
        # )
        #
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

        self.create_subscription(
            AckermannDriveStamped,
            f"/car{self.physical_ego_id}/cmd_vel",
            partial(self.cmd_callback, id=self.sim_ego_id),
            10,
        )
        self.create_subscription(
            AckermannDriveStamped,
            f"/car{self.physical_leader_id}/cmd_vel",
            partial(self.cmd_callback, id=self.sim_leader_id),
            10,
        )
        # subscription for radar data
        self.create_subscription(
            RadarFrame,
            "radar_data",
            self.radar_callback,
            # 10,
            qos_profile=qos_profile_sensor_data,
        )

        # self.model = PPO.load("./logs/01-02:11-16:48.89/best_model/best_model.zip")
        self.model = PPO.load("./logs/01-09:15-11:21.25/best_model/best_model.zip")

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
            self.draw = Draw(max_doppler, max_range, "Real Radar")
            self.draw_sim = Draw(max_doppler, max_range, "Sim Radar")
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

    def timer_callback(self):
        # if self.use_real_radar and self.real_obs is not None:
        #     action, _ = self.model.predict(self.real_obs)
        # else:
        #     action, _ = self.model.predict(self.obs)

        # self.obs, _, done, _, info = self.env.step(None)
        self.env.unwrapped._automatic_rendering()
        info = self.env.unwrapped._info()
        # done = False

        print(info)
        self.ttcs.append(info["ttc"])
        self.headways.append(info["headway"])

        # print(f"Sim: {info['leader_speed']}")
        # print(f"Sim: {self.env.unwrapped.road.vehicles[0].speed}")

        self.env.render()

    def pose_callback(self, msg, id):
        position = np.zeros(2)
        position[0] = msg.x
        position[1] = msg.y

        self.counter[id] += 1
        if self.counter[id] == 10:
            time_diff = time.time() - self.last_time[id]
            self.last_time[id] = time.time()

            vel = (position - self.last_pose[id]) / time_diff
            self.last_pose[id] = position
            speed = np.sqrt(vel[0] ** 2 + vel[1] ** 2)
            self.env.unwrapped.road.vehicles[id].speed = speed
            self.actual_speeds[id].append(speed)

            self.counter[id] = 0

        heading = msg.theta
        heading = highway_env.utils.wrap_to_pi(heading)
        self.env.unwrapped.road.vehicles[id].set_pose(position, heading)

    def cmd_callback(self, msg, id):
        self.commanded_speeds[id].append(msg.drive.speed)

    def publish(self, steering, speed, id):
        msg = AckermannDriveStamped()

        msg.drive.speed = speed
        msg.drive.steering_angle = np.rad2deg(steering)
        self.pubs[id].publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = CarPublisher(use_real_radar=True)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        np.save("ttc_end-to-end-bananaangle.npy", np.array(node.ttcs))
        np.save("headway_end-to-end-bananaangle.npy", np.array(node.headways))

        # np.save("commanded_speeds_ego.npy", np.array(node.commanded_speeds[0]))
        # np.save("commanded_speeds_leader.npy", np.array(node.commanded_speeds[1]))
        # np.save("actual_speeds_ego.npy", np.array(node.actual_speeds[0]))
        # np.save("actual_speeds_leader.npy", np.array(node.actual_speeds[1]))
        print("Interrupted")


if __name__ == "__main__":
    main()
