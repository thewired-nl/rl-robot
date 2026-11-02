from dataclasses import dataclass

from stable_baselines3.common.utils import LinearSchedule


class Subscriptable:
	def __getitem__(self, key):
		return getattr(self, key)


@dataclass
class DomainRandomization(Subscriptable):
	cmd_vel_range: tuple = (0.0, 0.0)
	cmd_yaw_range: tuple = (0.0, 0.0)
	cmd_zero_prob: float = 1.0
	cmd_resample_prob: float = 0.0

	action_delay_steps: int = 0
	action_delay_random: bool = False

	sensor_noise_std_dev: float = 0.0
	motor_noise_scale: float = 0.0

	push_prob: float = 0.00
	push_max_force: float = 0.0
	push_max_ticks: int = 0

	shift_weight_mm: float = 0.0
	mass_scale_range: tuple = (1.0, 1.0)
	friction_scale_range: tuple = (1.0, 1.0)
	motor_gain_range: tuple = (1.0, 1.0)
	ridge_prob: float = 0.0
	ridge_torque_max_nm: float = 0.0

	action_skip: float = 0.0


@dataclass
class Coefficients(Subscriptable):
	reward_alive: float = 1.0

	reward_yaw: float = 0.0
	reward_yaw_sigma: float = 0.5
	reward_vel: float = 0.0
	reward_vel_sigma: float = 0.1

	penalty_pitch: float = 0.0
	penalty_pitch_rate: float = 0.0
	# penalty_vel: float = 0.0
	# penalty_yaw: float = 0.0

	penalty_action: float = 0.0
	penalty_action_smoothness: float = 0.0
	penalty_position: float = 0.0

	thresh_tip: float = 30.0  # degrees
	thresh_spin: float = 4.0  # rad/s
	thresh_penalty: float = 2.0

	max_yaw_rate: float = 2.0  # rad/s
	max_vel: float = 0.5  # m/s


def configure(name, version):
	coef = Coefficients()
	dr = DomainRandomization()
	model_kwargs = dict(
		device="cpu",
		# n_steps=4096,  # per environment, per rollout
		n_steps=2048,  # per environment, per rollout
		# n_steps=128,
		# learning_rate=LinearSchedule(start=3e-4, end=0, end_fraction=1),
		learning_rate=0.0001995671586984195,
		gamma=0.9926920998378597,
		use_sde=False,
		# use_sde=True,
		# sde_sample_freq=4,
		policy_kwargs=dict(
			net_arch=[48, 48],
			optimizer_kwargs=dict(eps=1e-7),
		),
		clip_range_vf=1.0,
	)

	if version >= 1:
		dr.shift_weight_mm = 0.4
		coef.penalty_pitch = 0.5
		coef.penalty_action = 0.025
		if "actionskip" in name:
			dr.action_skip = 0.0005

	if version >= 2:
		if "actionskip" in name:
			dr.action_skip = 0.001
		dr.shift_weight_mm = 0.6
		coef.penalty_position = 0.01
		coef.penalty_pitch_rate = 0.02
		coef.reward_yaw = 1.0
		coef.reward_vel = 1.0
		coef.reward_vel_sigma = 0.1
		coef.reward_yaw_sigma = 0.5

	if version >= 3:
		if "cmd" in name:
			dr.cmd_vel_range = (-0.5, 0.5)
			dr.cmd_zero_prob = 0.5
			coef.max_vel = 0.5

		if "actionskip" in name:
			dr.action_skip = 0.0015

		coef.penalty_position = 0.02
		dr.shift_weight_mm = 1.0
		coef.penalty_action = 0.05
		coef.penalty_action_smoothness = 0.15
		coef.penalty_pitch_rate = 0.05
		# Prevent getting stuck in local maximum
		model_kwargs["ent_coef"] = 0.01  # encourage more exploration
		model_kwargs["clip_range_vf"] = 10.0  # allow more agressive critic updates

	if version >= 4:
		if "cmd" in name:
			dr.cmd_vel_range = (-1, 1)
			dr.cmd_zero_prob = 0.5
			dr.cmd_yaw_range = (-0.5, 0.5)
			coef.reward_yaw_sigma = 0.1
			coef.max_yaw_rate = 1.5

		if "actionskip" in name:
			dr.action_skip = 0.002

		coef.penalty_position = 0.025
		coef.max_vel = 0.5
		dr.shift_weight_mm = 2.0
		coef.penalty_action = 0.08
		dr.sensor_noise_std_dev = 0.01
		# coef.penalty_action_smoothness = 0.2
		dr.push_prob = 0.005
		dr.push_max_force = 0.2
		dr.push_max_ticks = 1
		dr.motor_noise_scale = 0.02

	if version >= 5:
		if "cmd" in name:
			dr.cmd_yaw_range = (-1, 1)
			dr.cmd_resample_prob = 0.005

		if "actionskip" in name:
			dr.action_skip = 0.004

		dr.shift_weight_mm = 3.0
		# dr.push_max_force = 0.5
		dr.push_max_ticks = 2
		dr.motor_noise_scale = 0.04
		dr.sensor_noise_std_dev = 0.025

	if version >= 6:
		coef.penalty_pitch_rate = 0.08
		dr.penalty_position = 0.03
		dr.mass_scale_range = (0.8, 1.2)
		dr.friction_scale_range = (0.5, 1.5)

	if version >= 7:
		coef.penalty_pitch_rate = 0.1
		dr.motor_gain_range = (0.8, 1.1)

	if version >= 8:
		dr.motor_gain_range = (0.5, 1.5)
		model_kwargs["learning_rate"] *= 1.5
		coef.penalty_action = 0.15
		coef.penalty_action_smoothness = 0.3

	if version >= 9:
		dr.shift_weight_mm = 0.5

	if version >= 10:
		coef.penalty_pitch_rate = 0.15
		coef.penalty_action_smoothness = 0.4
		coef.reward_yaw_sigma = 0.05
		coef.reward_vel_sigma = 0.05

	if version >= 11:
		model_kwargs["learning_rate"] *= 1.5
		coef.penalty_pitch_rate = 0.25
		coef.penalty_action_smoothness = 0.55
		coef.penalty_action = 0.05

	if version >= 12:
		if "cmd" in name:
			coef.max_vel = 0.9
			coef.max_yaw_rate = 3
		coef.penalty_position = 0.1
		coef.reward_yaw_sigma = 0.02
		coef.reward_vel_sigma = 0.02

	return dict(coef=coef, dr=dr, model_kwargs=model_kwargs)
