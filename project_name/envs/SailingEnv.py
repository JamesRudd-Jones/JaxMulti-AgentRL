import gymnasium as gym
from gymnax.environments import environment
from gymnax.environments import spaces
from math import sin, cos, radians
from pygame.math import Vector2
import os, random
from PIL import Image
from flax import struct
import jax
import jax.numpy as jnp
import jax.random as jrandom
import chex
from typing import Dict, Tuple, Any, Union, Optional
import time

import pygame
from pygame import gfxdraw


# class Wall:
#
#     def __init__(self, x1, y1, x2, y2):
#         self.x1 = x1
#         self.y1 = displayHeight - y1
#         self.x2 = x2
#         self.y2 = displayHeight - y2
#
#
#     def draw(self, surface):
#         pygame.draw.line(surface, (50,120,100), (self.x1, self.y1), (self.x2, self.y2), width=10)
#
#     """
#     returns true if the car object has hit this wall
#     """
#
#     def hitCar(self, car):
#         cw = car.width
#         # since the car sprite isn't perfectly square the hitbox is a little smaller than the width of the car
#         ch = car.height - 4
#         rightVector = Vector2(car.direction)
#         upVector = Vector2(car.direction).rotate(-90)
#         carCorners = []
#         cornerMultipliers = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
#         carPos = car.position
#         for i in range(4):
#             carCorners.append(carPos + (rightVector * cw / 2 * cornerMultipliers[i][0]) +
#                               (upVector * ch / 2 * cornerMultipliers[i][1]))
#
#         for i in range(4):
#             j = i + 1
#             j = j % 4
#             if linesCollided(self.x1, self.y1, self.x2, self.y2, carCorners[i].x, carCorners[i].y, carCorners[j].x,
#                               carCorners[j].y):
#                 return True
#         return False
#
#
# class RewardGate:
#
#     def __init__(self, x1, y1, x2, y2):
#         self.x1 = x1
#         self.y1 = y1
#         self.x2 = x2
#         self.y2 = y2
#         self.active = True
#
#         self.center = Vector2((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
#
#
#     def draw(self, surface):
#         if self.active:
#             pygame.draw.line(surface, (255,0,255), (self.x1, self.y1), (self.x2, self.y2), width=2)
#
#     """
#     returns true if the car object has hit this wall
#     """
#
#     def hitCar(self, car):
#         if not self.active:
#             return False
#
#         cw = car.width
#         # since the car sprite isn't perfectly square the hitbox is a little smaller than the width of the car
#         ch = car.height - 4
#         rightVector = Vector2(car.direction)
#         upVector = Vector2(car.direction).rotate(-90)
#         carCorners = []
#         cornerMultipliers = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
#         carPos = car.position
#         for i in range(4):
#             carCorners.append(carPos + (rightVector * cw / 2 * cornerMultipliers[i][0]) +
#                               (upVector * ch / 2 * cornerMultipliers[i][1]))
#
#         for i in range(4):
#             j = i + 1
#             j = j % 4
#             if linesCollided(self.x1, self.y1, self.x2, self.y2, carCorners[i].x, carCorners[i].y, carCorners[j].x,
#                               carCorners[j].y):
#                 return True
#         return False
#
#
# class Boat:
#     def __init__(self, x, y, walls, rewardGates, width=10, height=15):
#         self.x = x
#         self.y = y
#         self.position = Vector2(x, y)
#         self.speed = 0
#         self.angle = -30
#
#         self.direction = get_direction(self.angle)
#
#         self.dead = False
#         self.width = width
#         self.height = height
#         self.lineCollisionPoints = []
#         self.collisionLineDistances = []
#         self.vectorLength = displayWidth #length of vision vectors
#
#         self.turningLeft = False
#         self.turningRight = False
#         self.walls = walls
#         self.rewardGates = rewardGates
#         self.rewardNo = 0
#
#         self.directionToRewardGate = self.rewardGates[self.rewardNo].center - self.position
#
#         self.reward = 0
#
#         self.score = 0
#         self.lifespan = 0
#
#         self.steps_between_gate = 0
#         self.dist_between_gate = self.directionToRewardGate.length()
#
#         self.image = pygame.image.load("./boaty_boat.png").convert_alpha()
#         self.image = pygame.transform.scale(self.image, (width, height))
#         self.image_orig = self.image.copy()
#
#         self.max_speed = max_speed
#
#         self.max_length_on_screen = (displayHeight ** 2 + displayWidth ** 2) ** 0.5 #distance from corner to corner
#
#         self.last_distance_reward_gate = self.max_length_on_screen
#
#
#
#     def reset(self):
#         self.position = Vector2(self.x, self.y)
#         self.speed = 0
#         self.angle = -30
#
#         self.direction = get_direction(self.angle)
#
#         self.dead = False
#         self.lineCollisionPoints = []
#         self.collisionLineDistances = []
#
#         self.turningLeft = False
#         self.turningRight = False
#
#         self.rewardNo = 0
#         self.reward = 0
#
#         self.directionToRewardGate = self.rewardGates[self.rewardNo].center - self.position
#
#         self.steps_between_gate = 0
#         self.dist_between_gate = self.directionToRewardGate.length()
#
#         self.lifespan = 0
#         self.score = 0
#         for g in self.rewardGates:
#             g.active = True
#
#         self.last_distance_reward_gate = self.max_length_on_screen
#
#
#     def checkRewardGates(self):
#         self.reward = 0
#         if self.rewardGates[self.rewardNo].hitCar(self):
#             self.rewardGates[self.rewardNo].active = False
#             self.rewardNo += 1
#             self.score += 1
#             self.reward = 10000 * ((10000-self.steps_between_gate) / 10000) #10000 is the max steps per episode, this creates a time pressure
#             self.steps_between_gate = 0
#             if self.rewardNo == len(self.rewardGates):
#                 self.dist_between_gate = 100
#             else:
#                 self.directionToRewardGate = self.rewardGates[self.rewardNo].center - self.position
#                 self.dist_between_gate = self.directionToRewardGate.length()
#             #state = self.getState()
#             #print(state[-3])
#             if self.rewardNo == len(self.rewardGates):
#                 self.rewardNo = 0
#                 for g in self.rewardGates:
#                     g.active = True
#
#         self.directionToRewardGate = self.rewardGates[self.rewardNo].center - self.position
#
#         if self.last_distance_reward_gate > self.directionToRewardGate.length():
#             self.reward += ((self.dist_between_gate - self.directionToRewardGate.length()) / self.dist_between_gate)
#         elif self.last_distance_reward_gate < self.directionToRewardGate.length():
#             self.reward -= 1 - ((self.dist_between_gate - self.directionToRewardGate.length()) / self.dist_between_gate)
#
#         #print('{} last'.format(self.last_distance_reward_gate))
#         #print('{} now'.format(self.directionToRewardGate.length()))
#
#         self.last_distance_reward_gate = self.directionToRewardGate.length()
#
#     """
#     checks every wall and if the car has hit a wall returns true
#     """
#
#     def hitAWall(self):
#         for wall in self.walls:
#             if wall.hitCar(self):
#                 return True
#
#         return False
#
#
#     def getState(self):
#         self.setVisionVectors()
#         normalizedVisionVectors = [1 - (max(1.0, line) / self.vectorLength) for line in self.collisionLineDistances]
#         normalizedVisionVectors = [value + 1 if value == 0 else value for value in normalizedVisionVectors]
#         normalizedVisionVectors = [values if values >= 0.97 else 0 for values in normalizedVisionVectors]
#
#         normalizedAngleOfNextGate = (get_angle(self.direction) - get_angle(self.directionToRewardGate)) % 360
#         if normalizedAngleOfNextGate > 180:
#             normalizedAngleOfNextGate = -1 * (360 - normalizedAngleOfNextGate)
#
#         normalizedAngleOfNextGate /= 180
#
#         normalizedDirecToRewardGate_x = self.directionToRewardGate[0] / displayWidth
#         normalizedDirecToRewardGate_y = self.directionToRewardGate[1] / displayHeight
#
#         normalizedDistToRewardGate = self.directionToRewardGate.length() / self.max_length_on_screen
#
#         normalizedAngleToWind = angle_to_wind(self.angle) / np.pi
#
#         normalizedSpeed = self.speed / self.max_speed
#
#         normalizedState = [*normalizedVisionVectors, #0 to 1
#                            normalizedAngleOfNextGate,
#                            #normalizedDirecToRewardGate_x, #-1 to 1
#                            #normalizedDirecToRewardGate_y, #-1 to 1
#                            normalizedDistToRewardGate, #0 to 1
#                            normalizedAngleToWind, #0 to 1
#                            normalizedSpeed] #0 to 1
#         #print(normalizedState)
#         return np.array(normalizedState)
#
#
#     """
#     by creating lines in many directions from the car and getting the closest collision point of that line
#     we create  "vision vectors" which will allow the car to 'see'
#     kinda like a sonar system
#     """
#     def setVisionVectors(self):
#         h = self.height# - 4
#         w = self.width
#         self.collisionLineDistances = []
#         self.lineCollisionPoints = []
#         self.setVisionVector(w / 2, 0, 0)
#
#         self.setVisionVector(w / 2, -h / 2, -180 / 16)
#         self.setVisionVector(w / 2, -h / 2, -180 / 4)
#         #self.setVisionVector(w / 2, -h / 2, -4 * 180 / 8)
#
#         self.setVisionVector(w / 2, h / 2, 180 / 16)
#         self.setVisionVector(w / 2, h / 2, 180 / 4)
#         #self.setVisionVector(w / 2, h / 2, 4 * 180 / 8)
#
#         #self.setVisionVector(-w / 2, -h / 2, -6 * 180 / 8)
#         #self.setVisionVector(-w / 2, h / 2, 6 * 180 / 8)
#         #self.setVisionVector(-w / 2, 0, 180)
#
#
#
#     def getPositionOnCarRelativeToCenter(self, right, up):
#         rightVector = Vector2(self.direction)
#         rightVector.normalize()
#         upVector = self.direction.rotate(90)
#         upVector.normalize()
#
#         return self.position + ((rightVector * right) + (upVector * up))
#
#     """
#     returns the point of collision of a line (x1,y1,x2,y2) with the walls,
#     if multiple walls are hit it returns the closest collision point
#     """
#
#     def getCollisionPointOfClosestWall(self, x1, y1, x2, y2):
#         minDist = 2 * displayWidth
#         closestCollisionPoint = Vector2(0, 0)
#         for wall in self.walls:
#             collisionPoint = getCollisionPoint(x1, y1, x2, y2, wall.x1, wall.y1, wall.x2, wall.y2)
#             if collisionPoint is None:
#                 continue
#             if dist(x1, y1, collisionPoint.x, collisionPoint.y) < minDist:
#                 minDist = dist(x1, y1, collisionPoint.x, collisionPoint.y)
#                 closestCollisionPoint = Vector2(collisionPoint)
#         return closestCollisionPoint
#
#     """
#     calculates and stores the distance to the nearest wall given a vector
#     """
#
#     def setVisionVector(self, startX, startY, gamma):
#         collisionVectorDirection = self.direction.rotate(gamma)
#         collisionVectorDirection = collisionVectorDirection.normalize() * self.vectorLength
#         startingPoint = self.getPositionOnCarRelativeToCenter(startX, startY)
#         collisionPoint = self.getCollisionPointOfClosestWall(startingPoint.x, startingPoint.y,
#                                                               startingPoint.x + collisionVectorDirection.x,
#                                                               startingPoint.y + collisionVectorDirection.y)
#         if collisionPoint.x == 0 and collisionPoint.y == 0:
#             self.collisionLineDistances.append(self.vectorLength)
#         else:
#             self.collisionLineDistances.append(
#                 dist(startingPoint.x, startingPoint.y, collisionPoint.x, collisionPoint.y))
#         self.lineCollisionPoints.append(collisionPoint)
#     """
#     shows dots where the collision vectors detect a wall
#     """
#
#     def showCollisionVectors(self, surface):
#         for point in self.lineCollisionPoints:
#             if point != [0,0]:
#                 pygame.draw.line(surface, (131, 139, 139), (self.position.x, self.position.y), (point.x, point.y,), 1)
#                 pygame.draw.circle(surface, (0, 0, 0), (point.x, point.y), 5)
#
#
#     def updateWithAction(self, actionNo):
#         self.turningLeft = False
#         self.turningRight = False
#
#         if actionNo == 2:
#             self.turningLeft = True
#         elif actionNo == 1:
#             self.turningRight = True
#         elif actionNo == 0:
#             pass
#
#         totalReward = 0
#
#         for i in range(1):
#             if not self.dead:
#                 self.lifespan+=1
#                 self.steps_between_gate+=1
#                 self.move()
#
#                 if self.hitAWall():
#                     self.dead = True
#                     totalReward -= 100000
#                     # return
#
#                 #totalReward += self.reward
#
#                 if self.score == 32: #finishes game after 4 laps
#                     self.dead = True
#                     #totalReward += 100000
#
#                 self.checkRewardGates()
#                 totalReward += self.reward
#
#         self.setVisionVectors()
#
#         self.reward = totalReward
#
#         return self.getState(), self.reward, self.dead, {}
#
#
#
#     def move(self):
#
#         self.speed = self.rew(angle_to_wind(self.angle)) * 4
#
#         self.position.x = self.position.x - (self.speed * sin(radians(self.angle)))
#         self.position.y = self.position.y - (self.speed * cos(radians(self.angle)))
#
#         self.direction = get_direction(self.angle)
#
#         if self.turningRight:
#             self.angle -= 4
#         elif self.turningLeft:
#             self.angle += 4
#
#
#     def rew(self, theta, theta_0=0, theta_dead=np.pi / 12):
#         if angle_to_wind(self.angle) <= 7*np.pi/36:
#             return vel1(theta, theta_0, theta_dead) * np.cos(theta)
#         elif angle_to_wind(self.angle) > 7*np.pi/36 and angle_to_wind(self.angle) <= 5*np.pi/8:
#             return vel2(theta)
#         elif angle_to_wind(self.angle) > 5*np.pi/8 and angle_to_wind(self.angle) <= 3*np.pi/4:
#             return vel3(theta)
#         elif angle_to_wind(self.angle) > 3*np.pi/4 and angle_to_wind(self.angle) <= np.pi:
#             return vel4(theta)


@struct.dataclass
class EnvState(environment.EnvState):
    boat_pos: jnp.ndarray
    boat_dir: jnp.ndarray
    boat_dir_acc: jnp.ndarray
    boat_vel: jnp.ndarray
    boat_path: jnp.ndarray
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    max_steps_in_episode: int = 500
    wind_dir: float = 0.0
    wind_vel: jnp.ndarray = jnp.zeros(2)
    max_action: float = 0.001
    max_heading_vel: float = 1.0
    max_speed: float = 10.0
    acceleration: float = 1.0
    deceleration: float = 2.0
    dt: float = 1.0

    mass: float = 3000.0

    screen_width: int = 800
    screen_height: int = 600
    # boat_path_length: int = 30

    marks: jnp.ndarray = jnp.array(((400, 500),))
    # TODO to deal with multiple marks, could jnp.roll once done a conditional
    reward_gate: jnp.ndarray = jnp.array((10, 10))


class SailingEnv(environment.Environment[EnvState, EnvParams]):
    """
    0 degrees is the top of the screen or defined as north
    """
    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        params = EnvParams()
        params = params.replace(max_heading_vel=300.0 / 360.0 * 2 * jnp.pi * params.dt,    # radians per second
                                wind_vel=jnp.array((0.0, -50.0)) * params.dt)
        return params

    @staticmethod
    def vector_decomp(magnitude, angle):
        return magnitude * jnp.array((jnp.sin(angle), jnp.cos(angle)))

    @staticmethod
    def unit_vector(angle):
        return jnp.array((jnp.sin(angle), jnp.cos(angle)))

    @staticmethod
    def perpendicular(angle):
        return jnp.array((-angle[1], angle[0]))

    @staticmethod
    def angle_to_wind(heading, params):
        angle_diff = heading - params.wind_dir
        # Ensure the angle difference is between -pi and pi
        return (angle_diff + jnp.pi) % (2 * jnp.pi) - jnp.pi

    @staticmethod
    def angle_to_mark(state, params):
        abs_angle = jnp.arctan2(params.marks[0, 0] - state.boat_pos[0], params.marks[0, 1] - state.boat_pos[1])
        relative_angle = abs_angle - state.boat_dir
        normalised = (relative_angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
        return normalised

    @staticmethod
    def dist_to_mark(state, params):
        return params.marks[0] - state.boat_pos[0]

    def step_env(self, key: chex.PRNGKey, state: EnvState,  action: Union[int, float, chex.Array], params: EnvParams
                 ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        # 1. Update boat heading based on action
        action_1 = jnp.clip(action, -params.max_action, params.max_action)
        speed = jnp.dot(state.boat_vel, self.unit_vector(state.boat_dir))
        sqrtspeed = jax.lax.select(speed > 0,
                                   jnp.sqrt(jnp.linalg.norm(state.boat_vel)),
                                   -jnp.sqrt(jnp.linalg.norm(state.boat_vel)))
        new_boat_dir_acc = state.boat_dir_acc * 0.97  # TODO some decel modifier, maybe better way to state it
        new_boat_dir_acc = jnp.clip(new_boat_dir_acc + action_1.squeeze() * sqrtspeed,
                                      -params.max_heading_vel,
                                      params.max_heading_vel)
        new_heading = state.boat_dir + new_boat_dir_acc
        new_heading = jnp.mod(new_heading, 2 * jnp.pi)  # Wrap heading to be within 0 and 2*pi

        fcentripetal = new_boat_dir_acc * params.mass

        unit_heading_2 = self.unit_vector(new_heading)
        unit_perp_2 = self.perpendicular(unit_heading_2)

        # 2. Calculate the angle between the boat heading and wind direction.
        angle_diff = self.angle_to_wind(new_heading, params)

        # 3. Calculate the speed multiplier based on the polar curve.
        speed_multiplier = self.polar_curve(jnp.abs(angle_diff))  # TODO assuming polar curve is the same on both tacks
        apparent_wind_2 = params.wind_vel - state.boat_vel
        apparent_wind_speed = jnp.linalg.norm(apparent_wind_2)

        # 4. Update boat speed, accounting for acceleration/deceleration.
        SAILCOEFF = 7.0
        fdrive_2 = speed_multiplier * apparent_wind_speed * SAILCOEFF * unit_heading_2

        vforward_2 = jnp.dot(state.boat_vel, unit_heading_2) * unit_heading_2
        vperpendicular_2 = state.boat_vel - vforward_2

        fdrag_2 = -vforward_2 * jnp.linalg.norm(vforward_2) * 100.0  # opposite to direction of movement
        fkeel_2 = -vperpendicular_2 * jnp.linalg.norm(vperpendicular_2) * 1200.0
        fperp_2 = unit_perp_2 * fcentripetal * jnp.linalg.norm(state.boat_vel)

        new_boat_vel_2 = state.boat_vel + (fdrive_2 + fdrag_2 + fkeel_2 + fperp_2) / params.mass

        # 5. Update boat position based on heading and speed.
        new_boat_pos_2 = state.boat_pos + new_boat_vel_2 * params.dt

        # 6. Update boat path
        old_path = jnp.roll(state.boat_path, 1, axis=-1)
        new_boat_path = old_path.at[:, 0].set(new_boat_pos_2)

        # Update state dict and evaluate termination conditions
        new_state = EnvState(boat_pos=new_boat_pos_2,
                         boat_dir=new_heading,
                         boat_dir_acc=new_boat_dir_acc,
                         boat_vel=new_boat_vel_2,
                         boat_path=new_boat_path,
                         time=state.time + 1,
                         )

        reward = self.reward_func(state, new_state, params)

        done = self.is_terminal(new_state, params)

        # TODO same calcs are in get obs and reward and done, can we combine?

        return (jax.lax.stop_gradient(self.get_obs(new_state, params)),
                jax.lax.stop_gradient(new_state),
                jnp.array(reward),
                done,
                {"discount": self.discount(new_state, params)},
        )

    @staticmethod
    def polar_curve(theta):
        def vel(theta, theta_0=0, theta_dead=jnp.pi / 12):
            return 1 - jnp.exp(-(theta - theta_0) ** 2 / theta_dead)

        def rew(theta, theta_0=0, theta_dead=jnp.pi / 12):
            return vel(theta, theta_0, theta_dead) * jnp.cos(theta)

        def line_2(theta):
            return theta / (theta + 1) * 1.64

        def line_3(theta):
            return theta / (theta - 0.2) * 0.975

        def line_4(theta):
            return theta / (theta - 0.8) * 0.704

        boundaries = jnp.array([0, 7 * jnp.pi / 36, 5 * jnp.pi / 8, 3 * jnp.pi / 4, jnp.pi])
        functions = [rew, line_2, line_3, line_4]

        mask = (theta >= boundaries[:-1]) & (theta < boundaries[1:])

        result = jnp.sum(jnp.stack([jnp.where(mask, f(theta), 0) for mask, f in zip(mask, functions)]), axis=0)
        result = jnp.where(theta == boundaries[-1], functions[-1](theta), result)

        return result

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        # init_state = jrandom.uniform(key, minval=-0.05, maxval=0.05, shape=(4,))
        init_pos = jnp.array(((400.0,), (100.0,)))
        init_dir = jnp.radians(jnp.ones(1,) * 90)
        boat_speed = 1
        init_boat_vel = self.vector_decomp(boat_speed, init_dir)
        state = EnvState(boat_pos=init_pos.squeeze(axis=-1),
                         boat_dir=init_dir.squeeze(),
                         boat_dir_acc=jnp.zeros(1,).squeeze(),
                         boat_vel=init_boat_vel.squeeze(axis=-1),
                         # boat_path=jnp.repeat(init_pos, params.boat_path_length, axis=1),
                         boat_path=jnp.repeat(init_pos, 40, axis=1),
                         time=0,
                         )
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Applies observation function to state."""
        boat_speed = jnp.dot(state.boat_vel, self.unit_vector(state.boat_dir))
        angle_to_wind = self.angle_to_wind(state.boat_dir, params)
        angle_to_mark = self.angle_to_mark(state, params)
        dist_to_mark = self.dist_to_mark(state, params)
        obs = jnp.array([boat_speed,
                         angle_to_wind,
                         state.boat_dir_acc,
                         angle_to_mark,
                         jnp.linalg.norm(dist_to_mark),
                        ])
        return obs

    def reward_func(self, old_state: EnvState, state: EnvState, params: EnvParams) -> jnp.ndarray:
        done_x = jax.lax.select(jnp.logical_or(state.boat_pos[0] < 0, state.boat_pos[0] > params.screen_width),
                                jnp.array(True), jnp.array(False))
        done_y = jax.lax.select(jnp.logical_or(state.boat_pos[1] < 0, state.boat_pos[1] > params.screen_height),
                                jnp.array(True), jnp.array(False))
        done_boundaries = jnp.logical_or(done_x, done_y)
        done_time = jax.lax.select(state.time >= 3000, jnp.array(True), jnp.array(False))
        overall_done = jnp.logical_or(done_time, done_boundaries)
        # reward_dist = -jnp.linalg.norm(self.dist_to_mark(state, params), 8)#  / jnp.sqrt(jnp.square(params.screen_width) + jnp.square(params.screen_height))
        reward_dist = jnp.linalg.norm(self.dist_to_mark(old_state, params), 8) - jnp.linalg.norm(self.dist_to_mark(state, params), 8)
        reward = jax.lax.select(overall_done, -100.0, reward_dist)
        return reward

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        # """Check whether state is terminal."""
        dist_to_mark = self.dist_to_mark(state, params)
        done_dist = jax.lax.select(jnp.linalg.norm(dist_to_mark) <= 1, jnp.array(True), jnp.array(False))
        done_time = jax.lax.select(state.time >= 3000, jnp.array(True), jnp.array(False))
        done_x = jax.lax.select(jnp.logical_or(state.boat_pos[0] < 0, state.boat_pos[0] > params.screen_width),
                                jnp.array(True), jnp.array(False))
        done_y = jax.lax.select(jnp.logical_or(state.boat_pos[1] < 0, state.boat_pos[1] > params.screen_height),
                                jnp.array(True), jnp.array(False))
        done_boundaries = jnp.logical_or(done_x, done_y)
        done_inter = jnp.logical_or(done_dist, done_time)
        done = jnp.logical_or(done_boundaries, done_inter)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "SailingEnv-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(-params.max_action, params.max_action, (1,), dtype=jnp.float32)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        max_speed = 2
        max_dist = jnp.sqrt(jnp.square(params.screen_width) + jnp.square(params.screen_height))
        max_accel = 2.0
        # TODO sort out the above to be a bit better
        low = jnp.array([0.0,
                         -jnp.pi,
                         0.0,
                         -jnp.pi,
                         0.0,
                         ])
        high = jnp.array([max_speed,
                          jnp.pi,
                          max_accel,
                          jnp.pi,
                          max_dist,
                          ])
        return spaces.Box(-low, high, (5,), dtype=jnp.float32)

    def render(self, state: EnvState, params: EnvParams, render_delay: float = 0.0):
        """
        Remember width is left to right increasing
        BUT height is top to bottom increasing
        """
        if not hasattr(self, '_display'):
            pygame.init()
            self._display = pygame.display.set_mode((params.screen_width, params.screen_height))
            pygame.display.set_caption("Sailing Simulator")
        screen = self._display
        screen.fill((240, 240, 240))  # Light gray background

        # Convert state variables to screen coordinates
        def to_screen_coords(x, y):
            # x_offset = screen_width / 2
            # y_offset = screen_height / 2
            # scale = 50  # Adjust this scaling factor as needed
            # return (int(x_offset + x * scale), int((1 - y) * screen_height / 2))  # y is flipped in screen coords
            flip_y = (1 - (y / params.screen_height)) * params.screen_height
            return int(x), int(flip_y)

        # Draw Boat Path
        path_length = 30
        for i in range(path_length):
            p = state.boat_path[:, i]
            x, y = to_screen_coords(p[0], p[1])
            gfxdraw.aacircle(screen, x, y, 1, (0, 0, 255))
            gfxdraw.filled_circle(screen, x, y, 1, (0, 0, 255))

        # Draw Boat
        boat_angle = jnp.squeeze(state.boat_dir)
        boat_x_screen, boat_y_screen = to_screen_coords(state.boat_pos[0], state.boat_pos[1])

        # Load and rotate the boat image.
        boat_image_path = "project_name/envs/boaty_boat.png"
        boat_image = pygame.image.load(boat_image_path).convert_alpha()
        # Scale the image
        original_boat_width, original_boat_height = boat_image.get_size()
        boat_scale = 0.02
        new_boat_width = int(original_boat_width * boat_scale)
        new_boat_height = int(original_boat_height * boat_scale)
        scaled_boat_image = pygame.transform.scale(boat_image, (new_boat_width, new_boat_height))
        # Rotate the image.  pygame rotation is counter-clockwise, so we negate the angle.
        rotated_boat_image = pygame.transform.rotate(scaled_boat_image, -jnp.degrees(boat_angle))
        # Get the center of the rotated image.
        boat_rect = rotated_boat_image.get_rect(center=(boat_x_screen, boat_y_screen))
        # Blit the rotated image onto the screen.
        screen.blit(rotated_boat_image, boat_rect)

        boat_length = 10
        end_x = boat_x_screen + boat_length * jnp.sin(boat_angle)
        end_y = boat_y_screen - boat_length * jnp.cos(boat_angle)
        pygame.draw.line(screen, (255, 0, 0), (int(boat_x_screen), int(boat_y_screen)), (int(end_x), int(end_y)), 2)
        pygame.draw.circle(screen, (255, 0, 0), (int(boat_x_screen), int(boat_y_screen)), boat_length/2, 1)

        # Draw Wind Arrow
        wind_angle = params.wind_dir
        wind_length = 50
        wind_x = params.screen_width / 2
        wind_y = 0
        end_x = wind_x + wind_length * jnp.sin(wind_angle)
        end_y = wind_y + wind_length * jnp.cos(wind_angle)

        # Draw the wind arrow using gfxdraw for antialiasing
        pygame.draw.line(screen, (255, 0, 0), (wind_x, wind_y), (int(end_x), int(end_y)), 2)
        arrow_head_size = 10
        arrow_tip = (int(end_x), int(end_y))
        arrow_left = (int(end_x + arrow_head_size * jnp.sin(wind_angle - jnp.pi / 6)),
                      int(end_y - arrow_head_size * jnp.cos(wind_angle - jnp.pi / 6)))
        arrow_right = (int(end_x + arrow_head_size * jnp.sin(wind_angle + jnp.pi / 6)),
                       int(end_y - arrow_head_size * jnp.cos(wind_angle + jnp.pi / 6)))
        gfxdraw.aapolygon(screen, [arrow_tip, arrow_left, arrow_right], (255, 0, 0))
        gfxdraw.filled_polygon(screen, [arrow_tip, arrow_left, arrow_right], (255, 0, 0))
        # TODO check all the above actually works

        # Draw marks
        for i in range(params.marks.shape[0]):
            x, y = to_screen_coords(params.marks[i, 0], params.marks[i, 1])
            gfxdraw.aacircle(screen, x, y, 4, (0, 0, 0))
            gfxdraw.filled_circle(screen, x, y, 4, (0, 0, 0))

        # Draw Speed Text
        font = pygame.font.Font(None, 30)
        speed_in_fwd_dir = state.boat_vel[0] * jnp.sin(state.boat_dir) + state.boat_vel[1] * jnp.cos(state.boat_dir)
        speed_text = font.render(f"Speed: {jnp.squeeze(speed_in_fwd_dir):.2f} knots", True, (0, 0, 0))
        screen.blit(speed_text, (10, 10))

        # Draw Time Text
        time_text = font.render(f"Time: {state.time}", True, (0, 0, 0))
        screen.blit(time_text, (10, 40))

        # Draw Position Text
        pos_text = font.render(f"Position: ({state.boat_pos[0]:.2f}, {state.boat_pos[1]:.2f})", True, (0, 0, 0))
        screen.blit(pos_text, (10, 70))

        pygame.display.flip()
        pygame.event.pump()  # Process events to prevent freezing
        time.sleep(render_delay)


if __name__ == '__main__':
    with jax.disable_jit(disable=False):
        key = jrandom.PRNGKey(42)

        # Instantiate the environment & its settings.
        env = SailingEnv()
        env_params = env.default_params

        # Reset the environment.
        key, _key = jrandom.split(key)
        obs, state = env.reset(_key, env_params)

        time_steps = 3000  # 500
        # start_time = time.time()
        returns = 0
        for _ in range(time_steps):
            # Sample a random action.
            key, _key = jrandom.split(key)
            # action = env.action_space(env_params).sample(_key)
            # action = jnp.zeros(1,)
            action = jnp.ones(1,) * -0.001

            # env.render(state, env_params, render_delay=0.05)
            env.render(state, env_params)

            # Perform the step transition.
            key, _key = jrandom.split(key)
            obs, state, reward, done, _ = env.step(_key, state, action, env_params)
            returns += reward
            print(returns)
        # print(time.time() - start_time)

