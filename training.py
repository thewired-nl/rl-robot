import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "16"

import runner
import gym_env

num_cpu = 32

name = "clips-invert"
stage = 1


def configure(stage):
	coef = gym_env.Coefficients()
	dr = gym_env.DomainRandomization()
	model_kwargs = dict()

	if stage >= 1:
		coef.penalty_pitch = 0.5
		coef.penalty_action = 0.01

	if stage >= 2:
		coef.penalty_position = 0.01
		coef.penalty_pitch_rate = 0.02
		coef.reward_yaw = 1.0
		coef.reward_vel = 1.0

	if stage >= 2.1:
		coef.penalty_yaw = 0.01

	if stage >= 3:
		dr.cmd_vel_range = (-0.5, 0.5)
		dr.cmd_zero_prob = 0.5

		# Prevent getting stuck in local maximum
		model_kwargs["ent_coef"] = 0.01  # encourage more exploration
		model_kwargs["clip_range_vf"] = 10.0  # allow more agressive critic updates

	if stage >= 4:
		dr.cmd_vel_range = (-1, 1)
		dr.cmd_yaw_range = (-0.5, 0.5)
		coef.yaw_sigma = 0.1
		coef.max_yaw_rate = 1.5

	return (dict(coef=coef, dr=dr), model_kwargs)


if __name__ == "__main__":
	t = runner.TrainingCourse(name, configure, num_cpu, stage)
