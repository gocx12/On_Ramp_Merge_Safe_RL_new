import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from typing import Optional 
from typing import Tuple
Observation = np.ndarray

class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "forward_veh_distance_reward": 0,
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            controlled_vehicle = self.action_type.vehicle_class.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1) \
            + self.config["lane_change_reward"] * (action == 0 or action == 2) 
        reward = utils.lmap(reward,
                          [self.config["collision_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False

class HighwayEnv1Lane(HighwayEnv):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "lanes_count": 1,
        })
        return cfg
    
class HighwayEnv1LaneV1(HighwayEnv):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "lanes_count": 2,
            "vehicles_density": 2,
        })
        return cfg
    
class HighwayEnvV2(HighwayEnv):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "lanes_count": 2,
            "vehicles_density": 2,
            "collision_reward": -1,   
            "right_lane_reward": 0, 
            "high_speed_reward": 0.4,  
            "lane_change_reward": 0,  
            "reward_speed_range": [10, 20]
        })
        return cfg

class HighwayEnvV3(HighwayEnv):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "lanes_count": 2,
            "vehicles_density": 2,
            "collision_reward": -1,   
            "right_lane_reward": 0, 
            "high_speed_reward": 0.4,  
            "lane_change_reward": 0,  
            "reward_speed_range": [0, 20]
        })
        return cfg

class HighwayEnvV4(HighwayEnv):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "lanes_count": 2,
            "vehicles_density": 2,
            "collision_reward": -1,   
            "right_lane_reward": 0, 
            "high_speed_reward": 0.4,  
            "lane_change_reward": -1,  
            "reward_speed_range": [0, 10]
        })
        return cfg

class HighwayEnvV5(HighwayEnv):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "lanes_count": 2,
            "vehicles_density": 2,
            "collision_reward": -1,   
            "right_lane_reward": 0, 
            "high_speed_reward": 0.4,  
            "lane_change_reward": -1,  
            "reward_speed_range": [0, 10],
            "forward_veh_distance_reward": -0.01
        })
        return cfg
    
    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        
        forward_veh_dis_reciprocal = 0
        lane_index = self.vehicle.lane_index
        veh_front, veh_rear = self.road.neighbour_vehicles(self.vehicle)
        dis = 26
        if veh_front != None:
            dis = veh_front.position[0] - self.vehicle.position[0]
        if dis <= 25:
            forward_veh_dis_reciprocal = 1/dis

        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1) \
            + self.config["lane_change_reward"] * (action == 0 or action == 2) \
            + self.config["forward_veh_distance_reward"] * (abs(dis - 25)) 
        
        reward = utils.lmap(reward,
                          [self.config["collision_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward

class HighwayEnvV6(HighwayEnvV5):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "lanes_count": 2,
            "vehicles_density": 2,
            "collision_reward": -2,   
            "right_lane_reward": 0, 
            "high_speed_reward": 0.4,  
            "lane_change_reward": -1,  
            "reward_speed_range": [0, 10],
            "forward_veh_distance_reward": -0.01
        })
        return cfg
    def _reset(self) -> None:
        self._create_road()
        while True:
            self._create_vehicles()
            veh_front, veh_rear = self.road.neighbour_vehicles(self.vehicle)
            if veh_front != None:
                dis = veh_front.position[0] - self.vehicle.position[0]
            if dis > 20:
                break

class HighwayEnvV7(HighwayEnvV5):
    def _reset(self) -> None:
        self._create_road()
        while True:
            self._create_vehicles()
            veh_front, veh_rear = self.road.neighbour_vehicles(self.vehicle)
            if veh_front != None:
                dis = veh_front.position[0] - self.vehicle.position[0]
            if dis > 20:
                break

class HighwayEnvV8(HighwayEnvV5):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "lanes_count": 2,
            "vehicles_density": 1.5,
            "collision_reward": -2,   
            "right_lane_reward": 0, 
            "high_speed_reward": 0.4,  
            "lane_change_reward": -1,  
            "reward_speed_range": [0, 10],
            "forward_veh_distance_reward": -0.01
        })
        return cfg

class HighwayEnvV9(HighwayEnvV8):
    def step(self, action: Action, weight_action: Action) -> Tuple[Observation, float, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")
        
        self.steps += 1
        self._simulate(action, weight_action)

        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminal = self._is_terminal()
        info = self._info(obs, action)

        return obs, reward, terminal, info
    
    def _simulate(self, action: Optional[Action] = None, weight_action = None) -> None:
        traj = self.controlled_vehicles[0].get_traj(self.config["policy_frequency"], self.config["simulation_frequency"], action, weight_action)
        print("traj", traj)

        is_control = False

        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        for frame in range(frames):
            if is_control:
                # Forward action to the vehicle
                if action is not None \
                        and not self.config["manual_control"] \
                        and self.time % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                    self.action_type.act(action)
            else:
                # Forward action to the vehicle
                if action is not None \
                        and not self.config["manual_control"] :
                    print(traj, frame)
                    self.controlled_vehicles[0].act(traj, frame)

            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.time += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            if frame < frames - 1:  # Last frame will be rendered through env.render() as usual
                self._automatic_rendering()

        self.enable_auto_render = False


register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)

register(
    id='highway-fast-v0',
    entry_point='highway_env.envs:HighwayEnvFast',
)


register(
    id='highway-1lane-v0',
    entry_point='highway_env.envs:HighwayEnv1Lane',
)

register(
    id='highway-1lane-v1',
    entry_point='highway_env.envs:HighwayEnv1LaneV1',
)

register(
    id='highway-v2',
    entry_point='highway_env.envs:HighwayEnvV2',
)

register(
    id='highway-v3',
    entry_point='highway_env.envs:HighwayEnvV3',
)

register(
    id='highway-v4',
    entry_point='highway_env.envs:HighwayEnvV4',
)

register(
    id='highway-v5',
    entry_point='highway_env.envs:HighwayEnvV5',
)

register(
    id='highway-v6',
    entry_point='highway_env.envs:HighwayEnvV6',
)

register(
    id='highway-v7',
    entry_point='highway_env.envs:HighwayEnvV7',
)

register(
    id='highway-v8',
    entry_point='highway_env.envs:HighwayEnvV8',
)

register(
    id='highway-v9',
    entry_point='highway_env.envs:HighwayEnvV9',
)