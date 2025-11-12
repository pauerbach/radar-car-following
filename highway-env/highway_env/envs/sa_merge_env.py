import numpy as np
import scipy

import torch

TORCH_DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

from gymnasium.envs.registration import register

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import (
    RoadNetworkCommonRoad,
    RoadCommonRoad,
)
from highway_env.vehicle.behavior import ModelIDMVehicle, RandomVehicle

from commonroad.common.file_reader import CommonRoadFileReader


class SingleAgentMergeEnv(AbstractEnv):
    """
    A highway-env merge negotiation environment.

    The ego-vehicle is driving on a highway-env and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "action": {
                    "type": "ContinousSteering",
                },
                "observation": {"type": "Radar"},
                "duration": 50,  # time step
                # "policy_frequency": 40,  # [Hz]
                # "simulation_frequency": 40,  # [Hz]
                "policy_frequency": 10,  # [Hz]
                "simulation_frequency": 10,  # [Hz]
                "scaling": 320.76,
                "centering_position": [0.8, -0.6],
                # "termination_headway": 2.5,  # [m]
                "termination_headway": 4.5,  # [m]
                "speed_reward_weight": 2.0,
                "gap_reward_weight": 0.5,
                "jerk_reward_weight": 0.004,
                "minimum_gap": 2,
                "desired_time_gap": 1.5,
                "upper_time_gap": 10,
                "safety_weight": 0.5,
                "effiency_weight": 0.8,
            }
        )
        return cfg

    def set_vehicle(self, veh):
        self.vehicle = veh

    def _info(self):
        vehicle = self.vehicle
        leader = self.road.vehicles[1]

        # Safety criterion
        d = self._compute_headway_distance(vehicle)
        delta_v = vehicle.speed - leader.speed
        ttc = d / max(delta_v, 1e-6)  # clipping to ensure no division by zero

        # Driving effiency
        h = (d + leader.LENGTH) / vehicle.speed

        return {"ttc": ttc, "headway": h}

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions
        :param action: the action performed
        :return: the reward of the state-action transition
        """

        # return self._reward_li()
        original_reward = self._reward_li()
        mbrl_reward = self._reward_mbrl()

        # print(f"Original {original_reward} MBLR {mbrl_reward}")
        # assert abs(original_reward - mbrl_reward.item()) < 0.001
        # return self._reward_hart()

    def _reward_mbrl(self):
        obs = self.observation_type.observe()

        target_x = torch.tensor([obs[0]])
        target_y = torch.tensor([obs[1]])
        delta_v = -torch.tensor([obs[2]])
        ego_speed = torch.tensor([obs[3]])

        # Weights for the two components of the reward function
        SAFETY_WEIGHT = 0.5
        EFFICIENCY_WEIGHT = 0.8

        VEHICLE_LENGTH = 0.17

        # Create and move this tensor to GPU so that
        # we do not waste time moving it repeatedly to GPU later
        epsilon = torch.tensor([1e-6], device=TORCH_DEVICE, dtype=torch.double)
        two_pi = torch.tensor([2 * np.pi], device=TORCH_DEVICE, dtype=torch.double)

        # Safety criterion
        d = torch.hypot(target_x, target_y) - torch.tensor([VEHICLE_LENGTH])
        ttc = d / torch.maximum(
            delta_v, epsilon
        )  # clipping to ensure no division by zero

        r_safe = torch.where((ttc > 0.0) & (ttc < 1.5), torch.log(ttc / 1.5), 0.0)

        # Driving effiency
        mu = 0.4226
        sigma = 0.4365
        h = (d + VEHICLE_LENGTH) / ego_speed

        r_eff = (
            1
            # / (torch.sqrt(2 * torch.pi) * h * sigma)
            / (torch.sqrt(two_pi) * h * sigma)
            * torch.exp(-((torch.log(h) - mu) ** 2) / (2 * sigma**2))
        )

        return SAFETY_WEIGHT * r_safe + EFFICIENCY_WEIGHT * r_eff

    def _reward_li(self):
        """
        Reward function from Dianzhao from the paper
        Vision-based DRL Autonomous Driving Agent with Sim2Real Transfer
        which was used with the Duckiebots
        """

        vehicle = self.vehicle
        leader = self.road.vehicles[1]

        # Safety criterion
        d = self._compute_headway_distance(vehicle)
        delta_v = vehicle.speed - leader.speed
        ttc = d / max(delta_v, 1e-6)  # clipping to ensure no division by zero

        r_safe = 0
        if ttc > 0 and ttc < 1.5:
            r_safe = np.log(ttc / 1.5)

        # Driving effiency
        mu = 0.4226
        sigma = 0.4365
        h = (d + leader.LENGTH) / vehicle.speed
        h = np.maximum(h, 1e-6)

        r_eff = (
            1
            / (np.sqrt(2 * np.pi) * h * sigma)
            * np.exp(-((np.log(h) - mu) ** 2) / (2 * sigma**2))
        )

        return (
            self.config["safety_weight"] * r_safe
            + self.config["effiency_weight"] * r_eff
        )

    def _reward_hart(self):
        """
        Reward function from Fabian Hart from the paper
        Towards robust car-following based on deep reinforcement learning
        Same reward function is used in the paper
        Modified DDPG car-following model with a real-world human driving experience with CARLA simulator
        by Dianzhao
        """
        vehicle = self.vehicle
        leader = self.road.vehicles[1]

        # speed adherence (5)
        r_speed = 0
        if vehicle.speed > self.config["max_speed_reward"]:
            r_speed = -(vehicle.speed - self.config["max_speed_reward"])

        # jerk minimization (6)
        j_comf = 2  # [m/s**-3] TODO need to adjust this for model scale
        r_jerk = -(
            (1 / j_comf * vehicle.speed) ** 2
        )  # TODO need to calculate jerk instead of vehicle.speed

        # safety gap (7)
        b_comf = 2  # [m/s**2]
        v_dot_min = 9  # [m/s**2]
        b_kin = 0
        if vehicle.speed > leader.speed:
            leader_dist = self._compute_headway_distance(vehicle)
            b_kin = (vehicle.speed - leader.speed) ** 2 / (2 * leader_dist)

        r_safe = 0
        if b_kin > b_comf:
            r_safe = -np.tanh((b_kin - b_comf) / v_dot_min)

        # close gap to leader
        r_gap = 0
        g_t = self._compute_headway_distance(vehicle)
        g_min = self.config["minimum_gap"]
        T = self.config["desired_time_gap"]
        g_opt = vehicle.speed * T + g_min
        g_var = 0.5 * g_opt
        T_lim = self.config["upper_time_gap"]
        g_lim = vehicle.speed * T_lim + 2 * g_min
        g_star = 1.2 * g_opt  # TODO cant find value in paper

        r_gap = scipy.stat.norm.pdf((g_t - g_opt) / g_var) / scipy.stats.norm.pdf(0)
        if g_t > g_star:
            r_gap *= 1 - (g_t - g_star) / (g_lim - g_star)

        return (
            r_safe
            + self.config["speed_reward_weight"] * r_speed
            + self.config["gap_reward_weight"] * r_gap
            + self.config["jerk_reward_weight"] * r_jerk
        )

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs"""
        return (
            self.vehicle.crashed
            or self.steps >= self.config["duration"] * self.config["policy_frequency"]
            or self._compute_headway_distance(self.vehicle)
            > self.config["termination_headway"]
        )

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles(1, 1)
        self.T = int(self.config["duration"] * self.config["policy_frequency"])

    def _make_road(self) -> None:
        # scenario, _ = CommonRoadFileReader("./track.xml").open()
        # scenario, _ = CommonRoadFileReader("./track2.xml").open()
        # scenario, _ = CommonRoadFileReader("./rectangle_track.xml").open()
        scenario, _ = CommonRoadFileReader("./circular_track.xml").open()
        # scenario, _ = CommonRoadFileReader("./track3.xml").open()
        # scenario, _ = CommonRoadFileReader("./track5.xml").open()

        net = scenario.lanelet_network
        net_common_road = RoadNetworkCommonRoad(net, is_ring=True)

        self.road = RoadCommonRoad(network=net_common_road)

    def _make_vehicles(self, num_CAV=1, num_HDV=3) -> None:
        road = self.road

        ego_lane_index = 1
        leader_lane_index = 1

        initial_pos = np.random.rand()

        start_pos_cav = initial_pos
        start_pos_hdv = (
            initial_pos + RandomVehicle.LENGTH + np.random.rand() * 0.2
        )  # initial bumper gap [0, 0.2] m

        """Spawn CAV"""
        lane = self.road.network.get_lane(ego_lane_index)
        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            lane.position(start_pos_cav, 0),
            lane.heading_at(start_pos_cav),
            0.2,
        )

        self.vehicle = ego_vehicle
        self.vehicle.color = (200, 0, 150)
        self.vehicle.id = 0
        road.vehicles.append(ego_vehicle)

        """Spawn HDV"""
        lane = self.road.network.get_lane(leader_lane_index)
        veh = RandomVehicle(
            self.road,
            lane.position(start_pos_hdv, 0),
            lane.heading_at(start_pos_hdv),
            0.5,
        )

        road.vehicles.append(veh)


register(
    id="merge-single-agent-v0",
    entry_point="highway_env.envs:SingleAgentMergeEnv",
)
