from os import stat
from re import T
from typing import List, Tuple, Union
from unittest import result

import numpy as np
import copy
from highway_env import utils
from highway_env.road.road import Road, LaneIndex, Route
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.MPC_control import MPCSOLVE
from scipy.linalg import solve
import osqp
import scipy as sp
from scipy import sparse
import math

class ControlledVehicle(Vehicle):
    """
    A vehicle piloted by two low-level controller, allowing high-level actions such as cruise control and lane changes.

    - The longitudinal controller is a speed controller;
    - The lateral controller is a heading controller cascaded with a lateral position controller.
    """

    target_speed: float
    """ Desired velocity."""

    """Characteristic time"""
    TAU_ACC = 0.6  # [s]
    TAU_HEADING = 0.2  # [s]
    TAU_LATERAL = 0.6  # [s]

    TAU_PURSUIT = 0.5 * TAU_HEADING  # [s]
    KP_A = 1 / TAU_ACC
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 4  # [rad]
    DELTA_SPEED = 5  # [m/s]
    MAX_A = 5 # [m/s^2]

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: LaneIndex = None,
                 target_speed: float = None,
                 route: Route = None):
        super().__init__(road, position, heading, speed)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_speed = target_speed or self.speed
        self.route = route
        self.mpc = False
        self.future_state = None

    @classmethod
    def create_from(cls, vehicle: "ControlledVehicle") -> "ControlledVehicle":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed,
                target_lane_index=vehicle.target_lane_index, target_speed=vehicle.target_speed,
                route=vehicle.route)
        return v

    def plan_route_to(self, destination: str) -> "ControlledVehicle":
        """
        Plan a route to a destination in the road network

        :param destination: a node in the road network
        """
        try:
            path = self.road.network.shortest_path(self.lane_index[1], destination)
        except KeyError:
            path = []
        if path:
            self.route = [self.lane_index] + [(path[i], path[i + 1], None) for i in range(len(path) - 1)]
        else:
            self.route = [self.lane_index]
        return self

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action to change the desired lane or speed.

        - If a high-level action is provided, update the target speed and lane;
        - then, perform longitudinal and lateral control.

        :param action: a high-level action
        """
        if type(action) is dict or action == None:
            super().act(action)
            return

        self.follow_road()
        if action == "FASTER":
            self.target_speed += self.DELTA_SPEED
        elif action == "SLOWER":
            self.target_speed -= self.DELTA_SPEED
        elif action == "LANE_RIGHT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
        elif action == "LANE_LEFT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
        
        # if self.mpc:
        #     a, angle = self.mpc_control(self.target_speed, self.target_lane_index)
        #     a = a + (self.target_speed - self.speed)
        #     action = {'acceleration':a,'steering':angle}

        # else:
        action = {"steering": self.steering_control(self.target_lane_index),
                "acceleration": self.speed_control(self.target_speed)}

        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        action['acceleration'] = np.clip(action['acceleration'], -self.MAX_A, self.MAX_A)

        super().act(action)



    def test(self) :
        # sp.random.seed(1)
        m = 30
        n = 20
        Ad = sparse.random(m, n, density=0.7, format='csc')
        b = np.random.randn(m)

        # OSQP data
        P = sparse.block_diag([sparse.csc_matrix((n, n)), sparse.eye(m)], format='csc')
        q = np.zeros(n+m)
        A = sparse.bmat([[Ad,            -sparse.eye(m)],
                        [sparse.eye(n),  None]], format='csc')
        l = np.hstack([b, np.zeros(n)])
        u = np.hstack([b, np.ones(n)])

        # Create an OSQP object
        prob = osqp.OSQP()

        # Setup workspace
        prob.setup(P, q, A, l, u)

        # Solve problem
        res = prob.solve()

    def safe_act(self, action: Union[dict, str] = None) :
        self.follow_road()
        if action == "FASTER":
            self.target_speed += self.DELTA_SPEED
        elif action == "SLOWER":
            self.target_speed -= self.DELTA_SPEED
        elif action == "LANE_RIGHT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
        elif action == "LANE_LEFT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
        
        _,_ = self.mpc_control(self.target_speed, self.target_lane_index)

        return self.future_state

    def follow_road(self) -> None:
        """At the end of a lane, automatically switch to a next one."""
        if self.road.network.get_lane(self.target_lane_index).after_end(self.position):
            self.target_lane_index = self.road.network.next_lane(self.target_lane_index,
                                                                 route=self.route,
                                                                 position=self.position,
                                                                 np_random=self.road.np_random)

    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral speed command
        2. Lateral speed command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        target_lane = self.road.network.get_lane(target_lane_index)
        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.TAU_PURSUIT
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        # Lateral position control
        lateral_speed_command = - self.KP_LATERAL * lane_coords[1]
        # Lateral speed to heading
        heading_command = np.arcsin(np.clip(lateral_speed_command / utils.not_zero(self.speed), -1, 1))
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi/4, np.pi/4)
        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(heading_ref - self.heading)
        # Heading rate to steering angle
        steering_angle = np.arcsin(np.clip(self.LENGTH / 2 / utils.not_zero(self.speed) * heading_rate_command,
                                           -1, 1))
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        return float(steering_angle)

    def speed_control(self, target_speed: float) -> float:
        """
        Control the speed of the vehicle.

        Using a simple proportional controller.

        :param target_speed: the desired speed
        :return: an acceleration command [m/s2]
        """
        return self.KP_A * (target_speed - self.speed)

    def mpc_control(self, target_speed: float, target_lane_index: LaneIndex):
        '''
        input : reference veloctiy and target lane
        return: acceleration and steering angle used to control
        '''
        # get the reference
        v_ref = target_speed
        target_lane = self.road.network.get_lane(target_lane_index)
        local_position_ref = target_lane.local_coordinates(self.position)[0]
        position_ref = target_lane.position(local_position_ref, 0.)
        heading_ref = target_lane.heading_at(local_position_ref)
        current_steering = self.action['steering']
        car_info = dict(position=self.position, velocity=self.speed, heading=self.heading, steering_angle=current_steering)
        ref_info = dict(v_ref=v_ref,p_ref=position_ref,h_ref=heading_ref)

        # MPC setting
        rollout_N = 10
        mpc = MPCSOLVE(rollout_space=rollout_N)
        P, Q, G, H = mpc.compute_matrix(car_info,ref_info)

        # MPC solve
        mpc_result = mpc.solve(P=P, Q=Q, G=G, H=H)
        a = mpc_result[0,0]
        theta = mpc_result[1,0]

        # Predict the future ego position
        mpc_result = np.mat(mpc_result)
        future_N = rollout_N
        Y = mpc.A_cell * mpc.Esk + mpc.B_cell * mpc_result # error between reference and predict
        
        ## compute the reference state[x,y,v,heading]
        future_ref = np.mat(np.zeros([mpc.state_n * future_N, 1]))
        delta_dis = target_speed * 0.1
        for i in range(future_N):
            future_ref[i * mpc.state_n, 0] = target_lane.position(local_position_ref + delta_dis * (i+1), 0.)[0]
            future_ref[i * mpc.state_n + 1, 0] = target_lane.position(local_position_ref + delta_dis * (i+1), 0.)[1]
            future_ref[i * mpc.state_n + 2, 0] = target_speed
            future_ref[i * mpc.state_n + 3, 0] = target_lane.heading_at(local_position_ref + delta_dis * (i+1))
        
        self.future_state = future_ref + Y[0:future_N * mpc.state_n, 0]
        return a, theta

    def get_routes_at_intersection(self) -> List[Route]:
        """Get the list of routes that can be followed at the next intersection."""
        if not self.route:
            return []
        for index in range(min(len(self.route), 3)):
            try:
                next_destinations = self.road.network.graph[self.route[index][1]]
            except KeyError:
                continue
            if len(next_destinations) >= 2:
                break
        else:
            return [self.route]
        next_destinations_from = list(next_destinations.keys())
        routes = [self.route[0:index+1] + [(self.route[index][1], destination, self.route[index][2])]
                  for destination in next_destinations_from]
        return routes

    def set_route_at_intersection(self, _to: int) -> None:
        """
        Set the road to be followed at the next intersection.

        Erase current planned route.

        :param _to: index of the road to follow at next intersection, in the road network
        """

        routes = self.get_routes_at_intersection()
        if routes:
            if _to == "random":
                _to = self.road.np_random.randint(len(routes))
            self.route = routes[_to % len(routes)]

    def predict_trajectory_constant_speed(self, times: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """
        Predict the future positions of the vehicle along its planned route, under constant speed

        :param times: timesteps of prediction
        :return: positions, headings
        """
        coordinates = self.lane.local_coordinates(self.position)
        route = self.route or [self.lane_index]
        return tuple(zip(*[self.road.network.position_heading_along_route(route, coordinates[0] + self.speed * t, 0)
                     for t in times]))

class MDPVehicle(ControlledVehicle):

    """A controlled vehicle with a specified discrete range of allowed target speeds."""

    SPEED_COUNT: int = 10  # []
    SPEED_MIN: float = 0  # [m/s]
    SPEED_MAX: float = 30  # [m/s]

    def __init__(self,
                 road: Road,
                 position: List[float],
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: LaneIndex = None,
                 target_speed: float = None,
                 route: Route = None) -> None:
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.speed_index = self.speed_to_index(self.target_speed)
        self.target_speed = self.index_to_speed(self.speed_index)

    # def act(self, action: Union[dict, str] = None) -> None:
    #     """
    #     Perform a high-level action.

    #     - If the action is a speed change, choose speed from the allowed discrete range.
    #     - Else, forward action to the ControlledVehicle handler.

    #     :param action: a high-level action
    #     """
    #     if action == "FASTER":
    #         self.speed_index = self.speed_to_index(self.speed) + 1
    #     elif action == "SLOWER":
    #         self.speed_index = self.speed_to_index(self.speed) - 1
    #     else:
    #         super().act(action)
    #         return
    #     self.speed_index = int(np.clip(self.speed_index, 0, self.SPEED_COUNT - 1))
    #     self.target_speed = self.index_to_speed(self.speed_index)
    #     super().act()

    def act(self, traj=None, frame=None) -> None:
        """
        Perform a high-level action to change the desired lane or speed.
        :param action: a high-level action
        """
        if traj is None or frame is None:
            return

        W_Steer = 1
        W_Acc = 1

        # 找到轨迹中中最近的下一个点
        s_close = traj[0][frame]
        l_close = traj[1][frame]
        # for i in range(traj.shape[0]):
        #     if traj[i, 0] > self.position[0]:
        #         break
        
        # 弧度制
        theta = math.atan((l_close - self.position[1]) / (s_close - self.position[0]))
            
        dtheta = self.heading - theta
        # print("dtheta", dtheta)
        steering = dtheta / math.pi * 180 * W_Steer

        # print("traj[0]", traj[0], " self.position[0]", self.position[0])
        acceleration = (traj[0, frame] - self.position[0]) * W_Acc
        print("steering:", steering, " acceleration:", acceleration)

        steering = 0
        # acceleration = -10
        action = {"steering": steering,
                "acceleration": acceleration}
        print("control action:", action)

        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        action['acceleration'] = np.clip(action['acceleration'], -self.MAX_A, self.MAX_A)

        super().act(action)

    
    def get_traj(self, policy_frequency : int , simulation_frequency : int, action : Union[dict, str] = None):
        refer_line = self.get_refer_line(policy_frequency, simulation_frequency, action)
        print("refer_line:", refer_line, " position:", self.position)
        traj = self.traj_optim(refer_line)
        print("refer_line:", refer_line, "traj:", traj)
        return np.array([refer_line[0], traj])
        
        # traj --> action['acceleration'], action['steering']
        action['acceleration'] = 0
        action['steering'] = 0
        super().act(action)

    def get_refer_line(self, policy_frequency : int , simulation_frequency : int, action : Union[dict, str] = None) :
        self.follow_road()
        target_lane_index = self.target_lane_index
        
        ACTIONS_ALL = {
            0: 'LANE_LEFT',
            1: 'IDLE',
            2: 'LANE_RIGHT',
            3: 'FASTER',
            4: 'SLOWER'
        }
        action = ACTIONS_ALL[action]
        print("upper action",action)

        if action == "FASTER":
            self.target_speed += self.DELTA_SPEED
        elif action == "SLOWER":
            self.target_speed -= self.DELTA_SPEED
        elif action == "LANE_RIGHT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
        elif action == "LANE_LEFT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
        
        frames = int(simulation_frequency // policy_frequency)
        time = 1 / policy_frequency

        # 纵向参考轨迹用匀加速直线运动模型拟合
        acceleration = (self.target_speed - self.speed) / time

        # 横向参考轨迹用多项式拟合
        target_lane = self.road.network.get_lane(target_lane_index)
        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + (self.speed + acceleration*(1/policy_frequency)) * self.TAU_PURSUIT
        
        origin_pose = [self.position[0], self.position[1], self.heading]
        target_pose = [lane_next_coords, lane_coords[1], target_lane.heading_at(lane_next_coords)]        
        tri_curve = self.tri_curve(origin_pose, target_pose, policy_frequency)

        # 采样
        ds = np.linspace(0, target_pose[0]-origin_pose[0], frames)
        # print("ds:", ds)
        longitudes = origin_pose[0] * np.ones([len(ds)]) + ds # s-t
        latitudes = tri_curve[0] * ds**3 + tri_curve[1] * ds**2 + tri_curve[2] * ds + tri_curve[3] # s-l

        # print("longitudes:", longitudes)
        # print("latitudes:", latitudes)
        # print("position:", self.position)
        refer_line = np.array([longitudes, latitudes])
        return refer_line

    def tri_curve(self, origin:list, target:list, policy_frequency:int) :
        s = target[0] - origin[0]
        A = np.array([[0, 0, 0, 1],
                    [s**3, s**2, s, 1],
                    [0, 0, 1, 0],
                    [3*s**2, 2**s, 1, 0]])
        b = np.array([origin[1], target[1], origin[2], target[2]])
        # print("A:", A, " b:", b)
        x = solve(A, b)
        return x

    def traj_optim(self, refer_line):
        # 优化变量 x = [l0, l0', l0'', ..., ln, ln', ln''] 3*(len(refer_line))

        # Generate problem data
        # print("refer_line:", len(refer_line))
        n = len(refer_line[0])
        # Ad = sparse.random(n, density=0.5, format='csc')
        # x_true = np.random.randn(n) / np.sqrt(n)
        
        W_refer = 1

        # OSQP data
        Im = 2 * sparse.eye(n)
        Q = sparse.block_diag([Im], format='csc')
        f = - 2 * refer_line[1] * Q

        # 等式约束 Aeq * x = beq
        # 分段加加速度约束

        # 不等式约束 A * x <= b
        # 障碍约束
        Zn = sparse.csc_matrix(np.ones((n,n)))
        A = sparse.bmat([[Zn]], format='csc')
        l = np.hstack([-np.inf*np.ones(n)])
        u = np.hstack([np.inf*np.ones(n)])
        
        # Create an OSQP object
        prob = osqp.OSQP()
        # Setup workspace
        prob.setup(Q, f, A, l, u)

        # Solve problem
        res = prob.solve()
        print("qp res:", res)
        return np.array(res.x)


    def index_to_speed(self, index: int) -> float:
        """
        Convert an index among allowed speeds to its corresponding speed

        :param index: the speed index []
        :return: the corresponding speed [m/s]
        """
        if self.SPEED_COUNT > 1:
            return self.SPEED_MIN + index * (self.SPEED_MAX - self.SPEED_MIN) / (self.SPEED_COUNT - 1)
        else:
            return self.SPEED_MIN

    def speed_to_index(self, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - self.SPEED_MIN) / (self.SPEED_MAX - self.SPEED_MIN)
        return np.int(np.clip(np.round(x * (self.SPEED_COUNT - 1)), 0, self.SPEED_COUNT - 1))

    @classmethod
    def speed_to_index_default(cls, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - cls.SPEED_MIN) / (cls.SPEED_MAX - cls.SPEED_MIN)
        return np.int(np.clip(np.round(x * (cls.SPEED_COUNT - 1)), 0, cls.SPEED_COUNT - 1))

    @classmethod
    def get_speed_index(cls, vehicle: Vehicle) -> int:
        return getattr(vehicle, "speed_index", cls.speed_to_index_default(vehicle.speed))

    def predict_trajectory(self, actions: List, action_duration: float, trajectory_timestep: float, dt: float) \
            -> List[ControlledVehicle]:
        """
        Predict the future trajectory of the vehicle given a sequence of actions.

        :param actions: a sequence of future actions.
        :param action_duration: the duration of each action.
        :param trajectory_timestep: the duration between each save of the vehicle state.
        :param dt: the timestep of the simulation
        :return: the sequence of future states
        """
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # High-level decision
            for _ in range(int(action_duration / dt)):
                t += 1
                v.act()  # Low-level control action
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states
    
    def show_mpc_trajectory(self) -> List[ControlledVehicle]:
        '''
        show the mpc predict state
        '''
        states = []
        v = copy.deepcopy(self)
        for i in range(self.future_state.shape[0] // 4):
            if i == 4:
                v.position = [self.future_state[i*4, 0], self.future_state[i*4+1,0]]
                v.speed = self.future_state[i*4+2, 0]
                v.heading = self.future_state[i*4+3, 0]
                states.append(copy.deepcopy(v))
        
        return states