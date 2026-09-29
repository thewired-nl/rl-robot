from pathlib import Path
from types import SimpleNamespace
import math
import time

import gymnasium as gym
from gymnasium import spaces
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

import mujoco.viewer as mjv

from config import Coefficients, DomainRandomization


def get_mj_model(name="robot.xml") -> mj.MjModel:
	# load MuJoCo model
	cwd = Path(".")
	mujoco_file = cwd / "mujoco-setup" / name
	# mujoco_file = cwd / "mujoco-setup" / "robot-bala2.xml"
	if not mujoco_file.is_file():
		raise Exception(
			f"{mujoco_file} does not exist, run 00-mujoco-setup/import-urdf.ipynb first"
		)

	return mj.MjModel.from_xml_path(str(mujoco_file))


class BalanceEnv(gym.Env):
	render_sleep = True
	debug = False

	def __init__(
		self,
		coef: Coefficients,
		dr: DomainRandomization,
		max_steps=10_000,
		model_xml="robot.xml",
	):
		super().__init__()

		# two parameters, each motor from -1.0 to 1.0
		self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

		# [pitch, pitch_rate, yaw_rate, cmd_vel, cmd_yaw]
		obs_scale = np.array([np.pi, 20.0, 20.0, 1.0, 1.0], dtype=np.float32)
		self.observation_space = spaces.Box(
			low=-1 * obs_scale, high=obs_scale, dtype=np.float32
		)

		self.model = get_mj_model(model_xml)
		self.data = mj.MjData(self.model)

		self.m = SimpleNamespace(
			ground=self.model.geom("ground"),  # to modify friction
			robot=self.model.body("root"),  # to modify mass, apply forces
			# lipos=self.model.body("lipos"),  # to modify position offset
			# to modify gearing
			# act1=self.model.actuator("act_wheel_left"),
			# act2=self.model.actuator("act_wheel_right"),
		)

		self.d = SimpleNamespace(
			robot=self.data.body("root"),  # to get position
			act1=self.data.actuator("act_wheel_left"),
			act2=self.data.actuator("act_wheel_right"),
			imu_gyro=self.data.sensor("imu_gyro"),
			imu_quat=self.data.sensor("imu_quat"),
			priv_vel=self.data.sensor("priv_vel"),
			jvel1=self.data.sensor("jointvel_wheel_left"),
			jvel2=self.data.sensor("jointvel_wheel_right"),
		)

		# values before domain randomization
		# self._orig = SimpleNamespace(
		# 	mRobot=self.m.robot.mass,
		# 	fGround=self.m.ground.friction,
		# 	gAct1=self.m.act1.gear,
		# 	gAct2=self.m.act2.gear,
		# )

		if coef.thresh_tip > np.pi:
			coef.thresh_tip = math.radians(coef.thresh_tip)

		self.coef = coef
		self.dr = dr

		self._step = 0
		self._viewer = None
		self._step_start = 0
		self.max_steps = max_steps

		self.forces = []

		self.eulX = []
		self.eulY = []
		self.eulZ = []
		self.vels = []
		self.m1 = []
		self.m2 = []
		self.eulTicks = []

	def reset(self, seed=None, options=None):
		super().reset(seed=seed)
		self._step = 0
		self._prev_action = np.zeros(2, dtype=np.float32)

		# reset mujoco and run simulation forward without actually 'stepping' to fill sensor values
		mj.mj_resetData(self.model, self.data)

		# Impart an initial angular velocity around the y axis so the agent learns to recover
		# Note: qvel[4] = wy (rad/s)
		self.data.qvel[3] += self.np_random.uniform(-0.5, 0.5)
		# and x axis
		self.data.qvel[4] += self.np_random.uniform(-0.5, 0.5)

		mj.mj_forward(self.model, self.data)

		if self.dr.shift_weight_mm:
			shift = self.np_random.uniform(
				-self.dr.shift_weight_mm, self.dr.shift_weight_mm
			)
			self.m.lipos.pos[1] = shift / 1000.0  # mm to m

		self._pitch = 0.0
		self._pitch_rate = 0.0
		self._yaw = 0.0
		self._yaw_rate = 0.0

		self._cmd_vel = 0.0
		self._cmd_yaw = 0.0
		if self.np_random.random() >= self.dr.cmd_zero_prob:
			self._cmd_vel = float(self.np_random.uniform(*self.dr.cmd_vel_range))
			self._cmd_yaw = float(self.np_random.uniform(*self.dr.cmd_yaw_range))

		self._cmd_zero_pos = np.array([self.d.robot.xpos[0], self.d.robot.xpos[1]])

		# TODO: action delay

		self.forces = []

		# plotting
		self.eul = [0.0, 0.0, 0.0]
		self.vel_ = [0.0, 0.0, 0.0]
		self.vel_actual = 0

		return self._get_obs(), {}

	def _get_obs(self):
		# [pitch (q), pitch_rate (g), yaw_rate(g), cmd_vel, cmd_yaw]

		# accel_x, _, accel_z = self.data.sensor("imu_accel").data
		gyro = self.d.imu_gyro.data
		quat = self.d.imu_quat.data

		r = Rotation.from_quat(quat, scalar_first=True)

		eul = r.as_euler("xyz", degrees=False, suppress_warnings=True)

		# privileged velocity info relies on pitch, without added noise
		vel = self.d.priv_vel.data  # local frame
		vel_ = r.apply(vel)  # global frame
		self.vel_ = vel_

		self.vel_actual = vel_[0]

		# w, x, y, z = self.d.imu_quat.data
		# pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))

		self._pitch = eul[0]
		self._pitch_rate = gyro[0]
		self._yaw = eul[2]
		self._yaw_rate = gyro[2]

		# print(
		# 	f"EUL {math.degrees(eul[0]):.4f}, {math.degrees(eul[1]):.4f}, {math.degrees(eul[2]):.4f}"
		# )

		self.eul = eul

		if self.dr.sensor_noise_std_dev > 0.0:
			self._pitch += self.np_random.normal(0.0, self.dr.sensor_noise_std_dev)
			self._pitch_rate += self.np_random.normal(0.0, self.dr.sensor_noise_std_dev)
			# self._yaw += self.np_random.normal(0.0, self.dr.sensor_noise_std_dev)
			self._yaw_rate += self.np_random.normal(0.0, self.dr.sensor_noise_std_dev)

		obs = np.array(
			[
				self._pitch,
				self._pitch_rate,
				# self._yaw,
				# 0,
				self._yaw_rate,
				self._cmd_vel,
				self._cmd_yaw,
			],
			dtype=np.float32,
		)

		return obs

	def _rescale(self, num: float):
		# model works best with floats in -1 to 1 range
		# but our motor control is int -255 to 255
		r = round(num * 255) / 255
		return r

	def step(self, action: tuple[float, float]):
		self._step_start = time.time()
		# TODO: dr action delay?

		# action[0] = self._rescale(action[0])
		# action[1] = self._rescale(action[1])

		if self.dr.motor_noise_scale > 0:
			noise = self.np_random.uniform(
				-self.dr.motor_noise_scale, self.dr.motor_noise_scale, size=action.shape
			)
			action += noise

		action = np.clip(action, -1.0, 1.0)

		# self.d.act1.ctrl = action[0] * 8
		# self.d.act2.ctrl = action[1] * -8

		self.d.act1.ctrl = -action[0]  # invert
		# self.d.act1.ctrl = action[0]
		self.d.act2.ctrl = action[1]

		# TODO: dr forces
		if self.dr.push_prob > 0 and self.dr.push_max_ticks > 0:
			if self.np_random.random() < self.dr.push_prob:
				push = self.np_random.uniform(
					-self.dr.push_max_force, self.dr.push_max_force, size=3
				)
				push[2] = 0
				self.forces.append(
					SimpleNamespace(
						ticks=self.np_random.integers(self.dr.push_max_ticks),
						force=push,
					)
				)
				print("added push", self.forces)

		# apply forces
		for f in self.forces:
			f.ticks -= 1
			if f.ticks <= 0:
				continue
			mj.mj_applyFT(
				self.model,
				self.data,
				f.force,
				[0, 0, 0],
				[0, 0, 0],
				self.m.robot.id,
				self.data.qfrc_applied,
			)

		# TODO: dr wheel torque

		# TODO: dr command

		mj.mj_step(self.model, self.data)
		self._step += 1

		obs = self._get_obs()

		yaw_rate_target = self._cmd_yaw * self.coef.max_yaw_rate
		vel_target = self._cmd_vel * self.coef.max_vel

		# reward = self.coef.alive_bonus

		rewards = dict(
			alive=1,
			yaw=math.exp(
				-((self._yaw_rate - yaw_rate_target) ** 2) / self.coef.reward_yaw_sigma
			),
			vel=math.exp(
				-((self.vel_actual - vel_target) ** 2) / self.coef.reward_vel_sigma
			),
		)

		reward = 0.0

		for name in rewards:
			coef = self.coef[f"reward_{name}"]
			rewards[name] *= coef
			reward += rewards[name]

		penalties = dict(
			pitch=self._pitch**2,
			pitch_rate=self._pitch_rate**2,
			# vel=(
			# 	(self.vel_actual - vel_target) ** 2  # xvel
			# 	+ 0.5 * (self.vel_[1] ** 2)  # yvel
			# 	+ 0.5 * (self.vel_[2] ** 2)  # zvel
			# ),
			yaw=(self._yaw_rate - yaw_rate_target) ** 2,
			action=np.sum(action**2),
			action_smoothness=np.sum((action - self._prev_action) ** 2),
			position=0,
		)

		self._prev_action = action.copy()

		x_pos = self.d.robot.xpos[0] - self._cmd_zero_pos[0]
		y_pos = self.d.robot.xpos[1] - self._cmd_zero_pos[1]

		# FIXME for velocity position error origin reset
		if self._cmd_vel == 0:
			penalties["position"] = x_pos**2 + y_pos**2

		penalty = 0.0

		for name in penalties:
			coef = self.coef[f"penalty_{name}"]
			penalties[name] *= coef
			penalty += penalties[name]

		# Penalties
		# penalty += self.coef.penalty_pitch * self._pitch**2
		# penalty += self.coef.penalty_pitch_rate * self._pitch_rate**2

		# penalty += self.coef.penalty_yaw * abs(self._yaw)

		# penalty += self.coef.penalty_action * np.sum(action**2)
		# penalty += self.coef.penalty_action_smoothness * np.sum(
		# 	(action - self._prev_action) ** 2
		# )

		# x_pos = self.d.robot.xpos[0] - self._cmd_zero_pos[0]
		# y_pos = self.d.robot.xpos[1] - self._cmd_zero_pos[1]
		# FIXME: if vel_target == 0.0 and yaw_target == 0.0:
		# pos_penalty = self.coef.penalty_position * (x_pos**2 + y_pos**2)
		# penalty += pos_penalty

		# Gaussian reward: 1.0 for perfect tracking (no error) and decays smoothly toward 0 as error
		# grows. Sigma controls how quickly the reward decays with tracking error.
		# reward += self.coef.reward_vel * math.exp(
		# 	-((vel_actual - vel_target) ** 2) / self.coef.reward_vel_sigma
		# )
		# reward += self.coef.reward_yaw * math.exp(
		# 	-((self._yaw - yaw_target) ** 2) / self.coef.reward_yaw_sigma
		# )

		terminated = (
			abs(self._pitch) > self.coef.thresh_tip
		) is True  # convert np bool to python bool to satisfy check_env
		truncated = self._step >= self.max_steps

		if self.debug and self._step % 50 == 0:
			print(f"Step {self._step}")
			print(
				f"  pitch: {self._pitch:.3f} rad, pitch_rate: {self._pitch_rate:.3f} rad/s"
			)
			print(
				f"  yaw: {self._yaw:.3f} rad, yaw_rate: {self._yaw_rate:.3f} rad/s, target: {yaw_rate_target:.3f} (cmd {self._cmd_yaw:.3f})"
			)
			print(
				f"  vel: {self.vel_[0]:.3f} {self.vel_[1]:.3f} {self.vel_[2]:.3f} m/s"
			)
			print(
				f"  jvel: {self.d.jvel1.data[0]:.2f} {self.d.jvel2.data[0]:.2f} rad/s"
			)
			print(f"  actuators: [ {action[0]:.2f}, {action[1]:.2f} ]")
			print(f"  pos: [{x_pos:.3f}, {y_pos:.3f}] m")
			print(
				"  penalties:",
				", ".join(
					[f"{name}: {p:.4f}" for name, p in penalties.items() if p != 0]
				),
			)
			print(
				"  reward:",
				", ".join(
					[f"{name}: {r:.4f}" for name, r in rewards.items() if r != 0]
				),
			)
			print(
				f"  reward {reward:.3f}, penalty {penalty:.3f}, sum {(reward - penalty):.3f}"
			)
			print()

		return obs, (reward - penalty), terminated, truncated, {}

	def get_model(self):
		return self.model

	def get_data(self):
		return self.data

	def render(self):
		if self._viewer is None:
			self._viewer = mjv.launch_passive(self.model, self.data)
			# self._viewer.scn.flags[mj.mjtRndFlag.mjRND_SHADOW] = False

			self._viewer.cam.trackbodyid = self.d.robot.id
			self._viewer.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
			self._viewer.cam.lookat[:] = [0, 0, 0.05]
			self._viewer.cam.distance = 1
			self._viewer.cam.azimuth = 45
			self._viewer.cam.elevation = -25

		if self._step > 0:
			self.eulX.append(math.degrees(self.eul[0]))
			self.eulY.append(math.degrees(self.eul[1]))
			self.eulZ.append(math.degrees(self.eul[2]))
			self.m1.append(self._prev_action[0])
			self.m2.append(self._prev_action[1])
			self.vels.append(self.vel_actual)
			self.eulTicks.append(self._step)

		self._viewer.sync()

		if self.render_sleep:
			sleep = 0.005 - (time.time() - self._step_start)
			if sleep > 0:
				time.sleep(sleep)

	def plot(self):
		fig, ax1 = plt.subplots()
		ax1.plot(
			self.eulTicks, self.vels, label="forward velocity (m/s)", color="tab:purple"
		)
		ax1.set_ylim(-3, 3)
		ax2 = ax1.twinx()
		ax2.set_xlabel("step")
		ax2.set_ylabel("angle (deg)")
		ax2.plot(self.eulTicks, self.eulX, label="pitch", color="tab:red")
		ax2.plot(self.eulTicks, self.eulY, label="roll", color="tab:green")
		ax2.plot(self.eulTicks, self.eulZ, label="yaw", color="tab:blue")
		ax2.set_ylim(-180, 180)
		# ax1.set_ylabel("control")
		# ax1.plot(self.eulTicks, self.m1, label="motor 1")
		# ax1.plot(self.eulTicks, self.m2, label="motor 2")
		fig.tight_layout()
		ax1.legend(loc="upper left")
		ax2.legend(loc="upper right")
		plt.ioff()
		plt.show()

	def close(self):
		if self._viewer is not None:
			self._viewer.close()
			self._viewer = None
