import numpy as np
import mujoco as mj
import mujoco.viewer as mjv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper

import matplotlib.pyplot as plt

from gym_env import get_mj_model

import matplotlib

matplotlib.use("qtagg")

plt.ioff()
plt.rcParams["figure.raise_window"] = False


class VecRender(VecEnvWrapper):
	def __init__(self, venv: VecEnv):
		VecEnvWrapper.__init__(self, venv)

		# self._step = 0
		# self._viewer = None
		# self.model = get_mj_model()

	def step_wait(self):
		res = self.venv.step_wait()

		self.venv.env_method("render", indices=[0])

		# self._step += 1
		# if self._step % 20 != 0:
		# 	return res

		# data = self.venv.env_method("get_data", indices=[0])[0]
		# if self._viewer is None:
		# 	self._viewer = mjv.launch_passive(self.model, data)
		# 	self.data = data
		# 	self._viewer.cam.type = mj.mjtCamera.mjCAMERA_FREE
		# 	self._viewer.cam.lookat[:] = [0, 0, 0.05]
		# 	self._viewer.cam.distance = 0.8
		# 	self._viewer.cam.azimuth = 45
		# 	self._viewer.cam.elevation = -25

		# mj.mj_copyData(self.data, self.model, data)
		# self._viewer.sync()

		# self.renderer.update_scene(self.data, self.cam)

		return res

	def reset(self):
		res = self.venv.reset()
		return res

	def close(self):
		self.venv.close()
		super().close()
