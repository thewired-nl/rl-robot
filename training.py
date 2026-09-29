from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import (
	BaseCallback,
	EvalCallback,
	EveryNTimesteps,
	StopTrainingOnNoModelImprovement,
)

import gym_env


class TrainingSetup:
	def __init__(self, path: Path, cfg, mp, num_cpu=32):
		self.path = path
		self.cfg = cfg
		self.mp = mp
		self.num_cpu = num_cpu  # only used for training
		self.env_kwargs = dict(
			coef=self.cfg["coef"],
			dr=self.cfg["dr"],
			max_steps=self.cfg["model_kwargs"]["n_steps"],
			**self.cfg["env_kwargs"],
		)

	def load_model(self, env):
		kwargs = self.cfg["model_kwargs"] | dict(
			env=env, tensorboard_log=self.mp.tensorboard
		)

		if self.path.is_file():
			print("  loading existing model", self.path)
			return PPO.load(self.path, **kwargs)
		else:
			print("  creating new model")
			return PPO("MlpPolicy", **kwargs)

	def training_env(self):
		return make_vec_env(
			gym_env.BalanceEnv,
			n_envs=self.num_cpu,
			seed=42,
			env_kwargs=self.env_kwargs,
			vec_env_cls=SubprocVecEnv,
		)

	def eval_env(self):
		self._eval_env = gym_env.BalanceEnv(**self.env_kwargs)
		self._eval_env.render_mode = "human"

		return Monitor(env=self._eval_env)

	def eval_callback(self):
		return EvalCallback(
			self.eval_env(),
			best_model_save_path=self.mp.best.parent,
			deterministic=True,
			eval_freq=5000,
			callback_after_eval=StopTrainingOnNoModelImprovement(
				max_no_improvement_evals=12, min_evals=20, verbose=1
			),
		)

	class SaveProgress(BaseCallback):
		def __init__(self, mp):
			self._path = mp.progress
			super().__init__()

		def _on_step(self):
			_file = self._path(self.n_calls)
			print("saving model in-progress snapshot", _file)
			self.model.save(_file)
			return True

	def train(self, steps):
		model = self.load_model(self.training_env())
		model.learn(
			total_timesteps=steps,
			progress_bar=True,
			callback=[
				self.eval_callback(),
				EveryNTimesteps(10000 * self.num_cpu, self.SaveProgress(self.mp)),
			],
		)
		model.env.close()

	def eval(self, eps=3):
		eval_env = self.eval_env()
		model = self.load_model(eval_env)
		self._eval_env.debug = True
		evaluate_policy(
			model,
			eval_env,
			n_eval_episodes=eps,
			deterministic=True,
			render=True,
		)
		self._eval_env.plot()
		eval_env.close()
