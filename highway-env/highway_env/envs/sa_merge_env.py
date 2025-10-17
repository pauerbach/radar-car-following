import numpy as np

from gymnasium.envs.registration import register

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import (
    RoadNetworkCommonRoad,
    RoadCommonRoad,
)
from highway_env.vehicle.behavior import ModelIDMVehicle

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
                "duration": 15,  # time step
                "policy_frequency": 5,  # [Hz]
                "merging_speed_reward": -0.5,
                "right_lane_reward": 0.1,
                "lane_change_reward": -0.05,
                "reward_speed_range": [0.2, 0.6],
                "collision_reward": 200,
                "high_speed_reward": 1,
                "offramp_reward": 100,
                "HEADWAY_COST": 4,  # default=1
                "HEADWAY_TIME": 1.2,  # default=1.2[s]
                "MERGING_LANE_COST": 4,  # default=4
                "LANE_CHANGE_COST": 1,  # default=0.5
                "traffic_density": 1,  # easy or hard modes
                "scaling": 320.76,
                "centering_position": [0.8, -0.6],
            }
        )
        return cfg

    def set_vehicle(self, veh):
        self.vehicle = veh

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions
        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        vehicle = self.vehicle

        # TODO implement reward function from Dianzhaos paper
        return 0

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs"""
        return (
            self.vehicle.crashed
            or self.steps >= self.config["duration"] * self.config["policy_frequency"]
        )

    def _reset(self, num_CAV=1, num_HDV=6) -> None:
        self._make_road()
        self._make_vehicles(1, 1)
        self.T = int(self.config["duration"] * self.config["policy_frequency"])

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway-env and a merging lane.

        :return: the road
        """

        # scenario, _ = CommonRoadFileReader("./track.xml").open()
        scenario, _ = CommonRoadFileReader("./track2.xml").open()
        # scenario, _ = CommonRoadFileReader("./track3.xml").open()
        # scenario, _ = CommonRoadFileReader("./track5.xml").open()

        net = scenario.lanelet_network
        net_common_road = RoadNetworkCommonRoad(net, is_ring=True)

        self.road = RoadCommonRoad(network=net_common_road)

    def _make_vehicles(self, num_CAV=1, num_HDV=3) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """

        road = self.road

        ego_lane_index = 1
        leader_lane_index = 1
        start_pos_cav = 0.5
        start_pos_hdv = 1.5

        # initial speed with noise and location noise
        initial_speed = (
            np.random.rand(num_CAV + num_HDV) * 8 + 22
        )  # range from [25, 30]
        initial_speed /= 50  # scale to real model vehicles
        initial_speed = list(initial_speed)

        """Spawn CAV"""
        lane = self.road.network.get_lane(ego_lane_index)
        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            lane.position(start_pos_cav, 0),
            lane.heading_at(start_pos_cav),
            # initial_speed[0],
            0.5,
        )

        self.vehicle = ego_vehicle
        self.vehicle.color = (200, 0, 150)
        self.vehicle.id = 0
        road.vehicles.append(ego_vehicle)

        """Spawn HDV"""
        lane = self.road.network.get_lane(leader_lane_index)
        veh = ModelIDMVehicle(
            self.road,
            lane.position(start_pos_hdv, 0),
            lane.heading_at(start_pos_hdv),
            # initial_speed[1],
            1.0,
        )

        veh.enable_lane_change = False

        road.vehicles.append(veh)


register(
    id="merge-single-agent-v0",
    entry_point="highway_env.envs:SingleAgentMergeEnv",
)
