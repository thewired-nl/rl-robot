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
from kalman import KalmanFilter


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


# TODO: command history as way of smoothing outputs
# https://github.com/araffin/learning-to-drive-in-5-minutes/blob/master/donkey_gym/envs/vae_env.py#L51-L54
# The solution to smooth the control is to constrain the change in steering angle while augmenting the input with the history of previous commands (steering and throttle). That way, you impose continuity in the steering.


class BalanceEnv(gym.Env):
	# render_sleep = True
	debug = False
	debug_cmd = False

	def __init__(
		self,
		coef: Coefficients,
		dr: DomainRandomization,
		max_steps=10_000,
		model_xml="robot.xml",
		reward_norm=False,
		version=1,
		render_sleep=False,
	):
		super().__init__()

		self.version = version
		self.render_sleep = render_sleep

		# two parameters, each motor from -1.0 to 1.0
		self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

		# [pitch, pitch_rate, yaw_rate, enc1, enc2, cmd_vel, cmd_yaw, prev_act1, prev_act2]
		obs_scale = np.array(
			[np.pi, 20.0, 20.0, 50.0, 50.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
		)
		self.observation_space = spaces.Box(
			low=-1 * obs_scale, high=obs_scale, dtype=np.float32
		)

		self.reward_norm = reward_norm

		self.model = get_mj_model(model_xml)
		self.data = mj.MjData(self.model)

		self.m = SimpleNamespace(
			ground=self.model.geom("ground"),  # to modify friction
			robot=self.model.body("motor_base"),  # to modify mass, apply forces
			lipos=self.model.body("lipos"),  # to modify position offset
			elec=self.model.body("electronics"),  # to modify position offset
			# to modify gearing
			act1=self.model.actuator("act_wheel_left"),
			act2=self.model.actuator("act_wheel_right"),
		)

		self.d = SimpleNamespace(
			robot=self.data.body("root"),  # to get position
			act1=self.data.actuator("act_wheel_left"),
			act2=self.data.actuator("act_wheel_right"),
			imu_gyro=self.data.sensor("imu_gyro"),
			imu_accel=self.data.sensor("imu_accel"),
			imu_quat=self.data.sensor("imu_quat"),
			priv_vel=self.data.sensor("priv_vel"),
			jvel1=self.data.sensor("jointvel_wheel_left"),
			jvel2=self.data.sensor("jointvel_wheel_right"),
		)

		self.coef = coef
		self.dr = dr

		# values before domain randomization
		self._orig = SimpleNamespace(
			robot_mass=self.m.robot.mass,
			ground_friction=self.m.ground.friction,
			act1_force=self.m.act1.forcerange[1],
			act1_K=self.m.act1.gainprm[1],
			act2_force=self.m.act2.forcerange[1],
			act2_K=self.m.act2.gainprm[1],
			reward_alive=self.coef.reward_alive,
		)

		if coef.thresh_tip > np.pi:
			coef.thresh_tip = math.radians(coef.thresh_tip)

		self._step = 0
		self._viewer = None
		self._step_start = 0
		self.max_steps = max_steps

		self.forces = []

		self._action_delay = 0
		self._action_buffer = []

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

		# randomize main body mass
		if self.dr.mass_scale_range != (1.0, 1.0):
			scale = self.np_random.uniform(
				self.dr.mass_scale_range[0],
				self.dr.mass_scale_range[1],
			)
			# self.model.body_mass[self._chassis_id] = scale * self._chassis_mass_orig
			self.m.robot.mass = scale * self._orig.robot_mass

		# randomize friction
		if self.dr.friction_scale_range != (1.0, 1.0):
			scale = self.np_random.uniform(
				self.dr.friction_scale_range[0],
				self.dr.friction_scale_range[1],
			)
			self.m.ground.friction = scale * self._orig.ground_friction

		# randomize motor max torque
		if self.dr.motor_gain_range != (1.0, 1.0):
			for act in ["act1", "act2"]:
				scale = self.np_random.uniform(
					self.dr.motor_gain_range[0],
					self.dr.motor_gain_range[1],
				)
				scale2 = self.np_random.uniform(
					self.dr.motor_gain_range[0],
					self.dr.motor_gain_range[1],
				)
				K = getattr(self._orig, f"{act}_K") / scale
				force = getattr(self._orig, f"{act}_force") * scale2

				act = getattr(self.m, act)
				act.gainprm[1] = K
				act.forcerange = [-force, force]
				# print(
				# 	f"dr motor scale {scale:.2f} const {K:.2f}, force range {force:.2f}"
				# )

		# reset mujoco and run simulation forward without actually 'stepping' to fill sensor values
		mj.mj_resetData(self.model, self.data)

		# Impart an initial angular velocity around the x axis so the agent learns to recover
		# FIXME: disabled for static test
		self.data.qvel[3] += self.np_random.uniform(-0.5, 0.5)
		# and y axis
		self.data.qvel[4] += self.np_random.uniform(-0.5, 0.5)

		mj.mj_forward(self.model, self.data)

		if self.dr.shift_weight_mm:
			shift = self.np_random.uniform(
				-self.dr.shift_weight_mm, self.dr.shift_weight_mm
			)
			self.m.lipos.pos[1] = shift / 1000.0  # mm to m
			self.m.elec.pos[1] = (
				-shift / 1000.0
			)  # position is relative to lipo's, so offset back to its default position

		self.kalman_pitch = KalmanFilter(0.001, 0.003, 0.03)
		self._pitch = 0.0
		self._pitch_rate = 0.0
		self._yaw_rate = 0.0
		self._jvel = [0.0, 0.0]

		self._cmd_vel = 0.0
		self._cmd_yaw = 0.0

		if self.dr.cmd_zero_prob < 1:
			if self.np_random.random() >= self.dr.cmd_zero_prob:
				self._cmd_vel = float(self.np_random.uniform(*self.dr.cmd_vel_range))
				self._cmd_yaw = float(self.np_random.uniform(*self.dr.cmd_yaw_range))
				if self.debug_cmd:
					print(
						f"  random vel cmd {(self._cmd_vel * self.coef.max_vel):.3f} m/s, yaw cmd {(self._cmd_yaw * self.coef.max_yaw_rate):.3f} rad/s"
					)

		self._cmd_zero_pos = np.array([self.d.robot.xpos[0], self.d.robot.xpos[1]])

		if self.dr.action_delay_steps > 0 and self.dr.action_delay_random:
			self._action_delay = self.np_random.integers(
				0, self.dr.action_delay_steps + 1
			)
		else:
			self._action_delay = self.dr.action_delay_steps

		self._action_buffer = [
			np.zeros(self.action_space.shape) for _ in range(self._action_delay)
		]

		self.forces = []
		self.coef.reward_alive = self._orig.reward_alive

		# plotting
		self.eul = [0.0, 0.0, 0.0]
		self.vel_ = [0.0, 0.0, 0.0]
		self.vel_actual = 0

		return self._get_obs(), {}

	def _get_obs(self):
		# [pitch (rad), pitch_rate (rad/s), yaw_rate(rad/s), cmd_vel, cmd_yaw]

		# Quaternion is not provided by real IMU, so it + resulting euler should only be used as privileged info
		quat = self.d.imu_quat.data
		r = Rotation.from_quat(quat, scalar_first=True)
		self.eul = r.as_euler("xyz", degrees=False, suppress_warnings=True)

		_, accel_y, accel_z = self.d.imu_accel.data
		gyro = self.d.imu_gyro.data

		pitch_rate = gyro[0]
		accel_pitch = math.atan2(accel_y, accel_z)

		# Complementary filter
		alpha = 0.99
		self._pitch = (
			alpha * (self._pitch + pitch_rate * self.model.opt.timestep)
			+ (1 - alpha) * accel_pitch
		)

		# Kalman filter
		# self._pitch = self.kalman_pitch.update(
		# 	self._step * self.model.opt.timestep, accel_pitch, pitch_rate
		# )

		# From IMU 'perfect' quaternion
		# self._pitch = eul[0]

		# privileged velocity info relies on quaternion pitch, without added noise
		vel = self.d.priv_vel.data  # in local frame
		rYZ = Rotation.from_euler("x", self.eul[0], degrees=False)
		self.vel_ = rYZ.apply(vel)  # in world frame
		self.vel_actual = self.vel_[1]

		self._pitch_rate = gyro[0]
		self._yaw_rate = gyro[2]

		# 7 ppr encoders, 4x quadrature, 30:1 gearbox + 2:1 printed gears
		# ~48 pulses per 5ms at max speed (36rad/s)
		# miscounting 1 pulse is a difference of 0.75rad/s
		enc1 = self.d.jvel1.data[0]
		enc2 = self.d.jvel2.data[0]

		self._jvel = [enc1, enc2]

		# print(
		# 	f"EUL {math.degrees(eul[0]):.4f}, {math.degrees(eul[1]):.4f}, {math.degrees(eul[2]):.4f}"
		# )

		obs = np.array(
			[
				self._pitch,
				self._pitch_rate,
				self._yaw_rate,
				enc1,
				enc2,
				self._cmd_vel,
				self._cmd_yaw,
				self._prev_action[0],
				self._prev_action[1],
			],
			dtype=np.float32,
		)

		if self.dr.sensor_noise_std_dev > 0.0:
			obs[0] += self.np_random.normal(0.0, self.dr.sensor_noise_std_dev)
			obs[1] += self.np_random.normal(0.0, self.dr.sensor_noise_std_dev)
			obs[2] += self.np_random.normal(0.0, self.dr.sensor_noise_std_dev)
			obs[3] += self.np_random.normal(0.0, self.dr.sensor_noise_std_dev)
			obs[4] += self.np_random.normal(0.0, self.dr.sensor_noise_std_dev)
			# self._pitch += self.np_random.normal(0.0, self.dr.sensor_noise_std_dev)
			# self._pitch_rate += self.np_random.normal(0.0, self.dr.sensor_noise_std_dev)
			# enc1 += self.np_random.normal(0.0, self.dr.sensor_noise_std_dev)
			# enc2 += self.np_random.normal(0.0, self.dr.sensor_noise_std_dev)
			# self._yaw_rate += self.np_random.normal(0.0, self.dr.sensor_noise_std_dev)

		return obs

	def _rescale(self, num: float):
		# model works best with floats in -1 to 1 range
		# but our motor control is int -255 to 255
		r = round(num * 255) / 255
		return r

	def step(self, action: tuple[float, float]):
		self._step_start = time.time()

		self._prev_action = action.copy()

		if self.dr.motor_noise_scale > 0:
			noise = self.np_random.uniform(
				-self.dr.motor_noise_scale, self.dr.motor_noise_scale, size=action.shape
			)
			action += noise

		action = np.clip(action, -1.0, 1.0)

		if self.np_random.random() >= self.dr.action_skip:
			self.d.act1.ctrl = action[0] * self.m.act1.ctrlrange[1]
			self.d.act2.ctrl = action[1] * self.m.act1.ctrlrange[1]
		elif self.debug:
			print("actionskip")

		if self.dr.push_prob > 0 and self.dr.push_max_ticks > 0:
			if self.np_random.random() < self.dr.push_prob:
				push = self.np_random.uniform(
					-self.dr.push_max_force, self.dr.push_max_force, size=3
				)
				push[2] = 0

				# increase reward for staying alive after stronger pushes
				force = math.sqrt(np.sum(push**2))
				self.coef.reward_alive = self._orig.reward_alive * (
					1 + 0.2 * force / self.dr.push_max_force
				)

				self.forces.append(
					SimpleNamespace(
						ticks=self.np_random.integers(1, self.dr.push_max_ticks + 1),
						force=push,
					)
				)
				if self.debug:
					print(
						f"push [ {push[0]:4.3f} {push[1]:4.3f} ], force: {force:4.3f}, reward {self.coef.reward_alive:4.3f}"
					)

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
				self.m.elec.id,
				self.data.qfrc_applied,
			)

		self.forces = [f for f in self.forces if f.ticks > 0]

		if self.dr.cmd_resample_prob > 0:
			if self.np_random.random() < self.dr.cmd_resample_prob:
				if self.np_random.random() >= self.dr.cmd_zero_prob:
					self._cmd_vel = float(
						self.np_random.uniform(*self.dr.cmd_vel_range)
					)
					self._cmd_yaw = float(
						self.np_random.uniform(*self.dr.cmd_yaw_range)
					)
					if self.debug_cmd:
						print(
							f"  resample vel cmd {(self._cmd_vel * self.coef.max_vel):.3f} m/s, yaw cmd {(self._cmd_yaw * self.coef.max_yaw_rate):.3f} rad/s"
						)
				else:
					if self.debug_cmd:
						print("  resample vel/yaw zero")
					self._cmd_vel = 0
					self._cmd_yaw = 0
					self._cmd_zero_pos = [self.d.robot.xpos[0], self.d.robot.xpos[1]]

		mj.mj_step(self.model, self.data)
		self._step += 1

		obs = self._get_obs()
		# print(
		# 	f"[ {obs[0]:4.3f}, {obs[1]:4.3f}, {obs[2]:4.3f}, {obs[3]:4.3f}, {obs[4]:4.3f}, {obs[5]:4.3f}, {obs[6]:4.3f}, {obs[7]:4.3f}, {obs[8]:4.3f} ]"
		# )

		yaw_rate_target = self._cmd_yaw * self.coef.max_yaw_rate
		vel_target = self._cmd_vel * self.coef.max_vel

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
		reward_count = 0

		for name in rewards:
			coef = self.coef[f"reward_{name}"]
			if coef != 0:
				reward_count += 1
			rewards[name] *= coef
			reward += rewards[name]

		if self.reward_norm:
			reward = reward / reward_count

		penalties = dict(
			pitch=self._pitch**2,
			pitch_rate=self._pitch_rate**2,
			# vel=(
			# 	(self.vel_actual - vel_target) ** 2  # xvel
			# 	+ 0.5 * (self.vel_[1] ** 2)  # yvel
			# 	+ 0.5 * (self.vel_[2] ** 2)  # zvel
			# ),
			# yaw=(self._yaw_rate - yaw_rate_target) ** 2,
			action=np.sum(action**2),
			action_smoothness=np.sum((action - self._prev_action) ** 2),
			position=0,
		)

		x_pos = self.d.robot.xpos[0] - self._cmd_zero_pos[0]
		y_pos = self.d.robot.xpos[1] - self._cmd_zero_pos[1]

		if self._cmd_vel == 0:
			penalties["position"] = x_pos**2 + y_pos**2

		penalty = 0.0

		for name in penalties:
			coef = self.coef[f"penalty_{name}"]
			penalties[name] *= coef
			penalty += penalties[name]

		# convert np bool to python bool to satisfy check_env
		terminated = bool(abs(self._pitch) > self.coef.thresh_tip)

		# if abs(self._yaw_rate) > self.coef.thresh_spin:
		# 	terminated = True

		if penalty > self.coef.thresh_penalty:
			terminated = True

		truncated = self._step >= self.max_steps

		if self.debug and self._step % 50 == 0:
			print(f"Step {self._step}")
			print(
				f"  pitch: {self._pitch:.3f} rad, pitch_rate: {self._pitch_rate:.3f} rad/s"
			)
			print(
				f"  yaw: {self.eul[2]:.3f} rad, yaw_rate: {self._yaw_rate:.3f} rad/s, target: {yaw_rate_target:.3f} (cmd {self._cmd_yaw:.3f})"
			)
			print(
				f"  vel: {self.vel_actual:.3f} m/s ({self.vel_[0]:.3f} {self.vel_[1]:.3f} {self.vel_[2]:.3f}) m/s, target {vel_target:.3f} m/s (cmd {self._cmd_vel:.3f})"
			)
			print(f"  jvel: {self._jvel[0]:.2f} {self._jvel[1]:.2f} rad/s")
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
			# self._viewer = True
			self._viewer = mjv.launch_passive(self.model, self.data)

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
			sleep = self.model.opt.timestep - (time.time() - self._step_start)
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
