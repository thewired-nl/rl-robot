from dataclasses import dataclass

from stable_baselines3.common.utils import LinearSchedule


class Subscriptable:
	def __getitem__(self, key):
		return getattr(self, key)


@dataclass
class DomainRandomization(Subscriptable):
	sensor_noise_std_dev: float = 0.0
	# action_delay_steps: int = 0
	# action_delay_random: bool = False
	motor_noise_scale: float = 0.0
	push_prob: float = 0.00
	push_max_force: float = 0.0
	push_max_ticks: int = 0
	shift_weight_mm: float = 0.0
	# mass_scale_range: tuple = (1.0, 1.0)
	# friction_scale_range: tuple = (1.0, 1.0)
	# motor_gain_range: tuple = (1.0, 1.0)
	# ridge_prob: float = 0.0
	# ridge_torque_max_nm: float = 0.0
	cmd_vel_range: tuple = (0.0, 0.0)
	cmd_yaw_range: tuple = (0.0, 0.0)
	cmd_zero_prob: float = 0.0
	# cmd_resample_prob: float = 0.0


@dataclass
class Coefficients(Subscriptable):
	reward_alive: float = 1.0

	reward_yaw: float = 0.0
	reward_yaw_sigma: float = 0.5
	reward_vel: float = 0.0
	reward_vel_sigma: float = 0.1

	penalty_pitch: float = 0.0
	penalty_pitch_rate: float = 0.0
	penalty_vel: float = 0.0
	penalty_yaw: float = 0.0

	penalty_action: float = 0.0
	penalty_action_smoothness: float = 0.0
	penalty_position: float = 0.0

	#: float terminates run
	thresh_tip: float = 30.0  # degrees

	#: float commands
	max_yaw_rate: float = 2.0  # rad/s
	max_vel: float = 0.4  # m/s


def configure(name, version):
	coef = Coefficients()
	dr = DomainRandomization()
	model_kwargs = dict(
		device="cpu",
		# n_steps=4096,  # per environment, per rollout
		n_steps=2048,  # per environment, per rollout
		learning_rate=LinearSchedule(start=3e-4, end=0, end_fraction=1),
		use_sde=False,
		# use_sde=True,
		# sde_sample_freq=4,
		policy_kwargs=dict(
			net_arch=[48, 48],
			optimizer_kwargs=dict(eps=1e-5),
		),
		clip_range_vf=1.0,
	)

	if version >= 1:
		coef.penalty_pitch = 0.5
		coef.penalty_action = 0.01
		# coef.penalty_yaw = 0.005
		# coef.penalty_vel = 0.025

	if version >= 2:
		coef.penalty_position = 0.01
		coef.penalty_pitch_rate = 0.02
		coef.reward_yaw = 1.0
		coef.reward_vel = 1.0
		coef.reward_vel_sigma = 0.1
		coef.reward_yaw_sigma = 0.5
		coef.max_vel = 0.5

	# if version >= 2.1:
	# 	coef.penalty_yaw = 0.01

	if version >= 3:
		dr.cmd_vel_range = (-0.5, 0.5)
		dr.cmd_zero_prob = 0.5

		# Prevent getting stuck in local maximum
		model_kwargs["ent_coef"] = 0.01  # encourage more exploration
		model_kwargs["clip_range_vf"] = 10.0  # allow more agressive critic updates

	if version >= 4:
		dr.cmd_vel_range = (-1, 1)
		dr.cmd_yaw_range = (-0.5, 0.5)
		coef.yaw_sigma = 0.1
		coef.max_yaw_rate = 1.5

	return dict(coef=coef, dr=dr, model_kwargs=model_kwargs)
