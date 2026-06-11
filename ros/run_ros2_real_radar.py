import sys
from functools import partial

import numpy as np
import time

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Pose2D
from radar_msgs.msg import RadarFrame
from rclpy.qos import qos_profile_sensor_data

import cv2 as cv


sys.path.append("./highway-env/")

from highway_env.envs.common.observation import RadarPipeline, Draw

import highway_env
import gymnasium as gym
from stable_baselines3 import PPO


def cfar_2d(rd_map, guard_cells=2, training_cells=8, threshold_scale=1.5):
    detections = []

    rows, cols = rd_map.shape

    for r in range(training_cells + guard_cells, rows - training_cells - guard_cells):
        for d in range(
            training_cells + guard_cells, cols - training_cells - guard_cells
        ):
            # CUT = Cell Under Test
            cut = rd_map[r, d]

            noise_cells = []

            for i in range(
                r - training_cells - guard_cells, r + training_cells + guard_cells + 1
            ):
                for j in range(
                    d - training_cells - guard_cells,
                    d + training_cells + guard_cells + 1,
                ):
                    # Skip guard cells + CUT
                    if abs(i - r) <= guard_cells and abs(j - d) <= guard_cells:
                        continue

                    noise_cells.append(rd_map[i, j])

            noise_level = np.mean(noise_cells)
            threshold = threshold_scale * noise_level

            if cut > threshold:
                detections.append((r, d, cut))

    return detections


def detections_to_measurements(detections, range_res, doppler_res):
    measurements = []

    for r_idx, d_idx, power in detections:
        distance = r_idx * range_res
        velocity = d_idx * doppler_res

        measurements.append(
            {"distance": distance, "velocity": velocity, "power": power}
        )

    return measurements


def select_lead_vehicle(measurements):
    if not measurements:
        return None

    # Closest object (simplest assumption)
    lead = min(measurements, key=lambda x: x["distance"])
    return lead


class KalmanFilterCV:
    def __init__(self, dt, process_var=1.0, meas_var=1.0):
        self.dt = dt

        # State: [distance, velocity]
        self.x = np.zeros((2, 1))

        # State transition
        self.F = np.array([[1, dt], [0, 1]])

        # Measurement matrix
        self.H = np.eye(2)

        # Covariances
        self.P = np.eye(2) * 10
        self.Q = np.eye(2) * process_var
        self.R = np.eye(2) * meas_var

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.array(z).reshape(2, 1)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

    def get_state(self):
        return self.x.flatten()


## Taken from https://github.com/PawanGu/radar-camera-fusion-cfar/blob/main/notebooks/radar_signal_chain_cfar.py
def ca_cfar_2d(power_map, guard=1, train=4, pfa=1e-3):
    """
    2D Cell-Averaging CFAR on linear power map.
    guard: guard cells on each side (square guard region)
    train: training cells thickness around guard
    pfa: desired probability of false alarm
    Returns boolean detection map of same shape.
    """
    H, W = power_map.shape
    half = guard + train  # half window radius including guard+train

    # Integral image for fast rectangular sums
    I = np.pad(power_map, ((1, 0), (1, 0)), mode="constant")
    I = np.cumsum(np.cumsum(I, axis=0), axis=1)

    def rect_sum(y0, x0, y1, x1):
        return I[y1 + 1, x1 + 1] - I[y0, x1 + 1] - I[y1 + 1, x0] + I[y0, x0]

    det = np.zeros_like(power_map, dtype=bool)
    N_train = (2 * half + 1) ** 2 - (2 * guard + 1) ** 2
    alpha = N_train * (pfa ** (-1.0 / max(N_train, 1)) - 1.0)

    for y in range(half, H - half):
        for x in range(half, W - half):
            y0, x0 = y - half, x - half
            y1, x1 = y + half, x + half
            total_sum = rect_sum(y0, x0, y1, x1)

            yg0, xg0 = y - guard, x - guard
            yg1, xg1 = y + guard, x + guard
            guard_sum = rect_sum(yg0, xg0, yg1, xg1)

            noise_sum = total_sum - guard_sum
            noise_mean = noise_sum / max(N_train, 1)
            threshold = alpha * noise_mean

            if power_map[y, x] > threshold:
                det[y, x] = True
    return det


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

        self.max_distance = (
            self.env.unwrapped.observation_type.radar_simulator.get_max_range()
        )

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
        self.model = PPO.load(
            "./logs/01-09:15-11:21.25/best_model/best_model.zip"
        )  # End to End
        # self.model = PPO.load( "./logs/02-02:18-24:45.37/best_model/best_model.zip")  # direct input
        # self.model = PPO.load("./logs/02-17:14-05:37.57/best_model/best_model.zip")

        # self.model = PPO.load("./logs/02-26:16-57:07.77/best_model/best_model.zip")

        self.sim_timer = self.create_timer(
            1 / self.env.unwrapped.config["policy_frequency"], self.timer_callback
        )

        self.kf = KalmanFilterCV(dt=0.1)

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

        ################
        ## End to end
        ###############
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

        ################
        ## ChatGPT suggestion CFAR plus Kalmann
        ###############
        # detections = cfar_2d(
        #     np.average(doppler_fft, 2), guard_cells=4, training_cells=3
        # )
        #
        # # 2. Convert to physical measurements
        # c = 3e8  # Speed of light (m/s)
        # B = 5.36e9  # Bandwidth
        # fc = 60.75e9  # center frequency
        # T_chirp = 0.00123371  # chirp time
        # N_c = 32  # number of chirps
        #
        # lambda_radar = c / fc
        #
        # range_res = (
        #     c / (2 * B) / 2
        # )  # division by two is necessary to have the same position as in the original RD map
        # doppler_res = lambda_radar / (2 * N_c * T_chirp) / 2
        # measurements = detections_to_measurements(detections, range_res, doppler_res)
        # print(measurements)
        #
        # # 3. Select lead vehicle
        # lead = select_lead_vehicle(measurements)
        #
        # # 4. Kalman filter
        # self.kf.predict()
        #
        # if lead is not None:
        #     z = [lead["distance"], lead["velocity"]]
        #     # print(z)
        #     self.kf.update(z)
        #
        # kf_state = self.kf.get_state()
        # self.real_obs = np.array([kf_state[1], kf_state[0]])

        ################
        ## custom implemenatation CFAR plus Kalmann
        ###############
        # detections = ca_cfar_2d(np.average(doppler_fft, 2), guard=4, train=3, pfa=1e-3)
        # detections = detections.astype(np.uint8) * 255
        # # self.draw.draw(detections)
        #
        # contours, hierarchy = cv.findContours(
        #     detections,
        #     cv.RETR_LIST | cv.RETR_EXTERNAL,
        #     cv.CHAIN_APPROX_SIMPLE,
        # )
        # det_speed = det_dist = 0
        # if len(contours) > 0:
        #     cnt = max(contours, key=cv.contourArea)
        #     (x, y), radius = cv.minEnclosingCircle(cnt)
        #     w, h = detections.shape
        #     det_speed = (x - w / 2) / (w / 2)
        #     det_dist = y * (self.max_distance / 2) / h
        #     # det_dist = y * 0.9 / h
        #     # print(f"Center {(x-w/2)/(w/2)},{y*0.9/h}")
        #     # center = (int(x), int(y))
        #     # radius = int(radius)
        #     # detections = cv.cvtColor(detections, cv.COLOR_GRAY2RGB)
        #     # cv.circle(detections, center, radius, (0, 255, 0), 2)
        #     # print()
        #
        # self.kf.predict()
        # z = [det_dist, det_speed]
        # self.kf.update(z)
        # kf_state = self.kf.get_state()
        #
        # # self.real_obs = np.array([kf_state[1], kf_state[0]])
        # self.real_obs = self.env.unwrapped.observation_type.normalize_obs(kf_state)

        if self.use_real_radar and self.real_obs is not None:
            action, _ = self.model.predict(self.real_obs)
        else:
            action, _ = self.model.predict(self.obs)

        # self.obs, _, _, _, _ = self.env.step(action)
        self.env.unwrapped._simulate(action)

        self.env.render()

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
