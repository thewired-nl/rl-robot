import importlib
from pathlib import Path
import math

from stable_baselines3 import PPO
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import (
	BaseCallback,
	EvalCallback,
	EveryNTimesteps,
	StopTrainingOnNoModelImprovement,
	# StopTrainingOnRewardThreshold,
)

import gym_env
import render_env

importlib.reload(gym_env)
importlib.reload(render_env)


class TrainingCourse:
	_models_path = Path("models")
	model = None
	vec_env = None
	_eval_env = None

	def __init__(self, name, configure, num_cpu=1, stage=1):
		self._name = name
		self.stage = stage
		self.update_name()
		self.num_cpu = num_cpu

		self.configure = configure
		self.update_config()
		# self.setup_envs()
		self.get_model()

	def update_config(self):
		kwargs, model_kwargs = self.configure(self.stage)
		self.kwargs = kwargs
		self.model_kwargs = model_kwargs | dict(tensorboard_log=str(self.log_path))
		self.setup_envs()
		print("config: updated env kwargs")
		if self.model is not None:
			self.model.__dict__.update(self.model_kwargs)
			print("config: updated model kwargs")

	def update_name(self):
		self.name = f"{self._name}-stage{self.stage}"
		self.runs_path = self._models_path / self.name / "runs"
		self.model_path = self._models_path / self.name / "model.zip"
		self.best_model_path = self.runs_path / "best_model.zip"
		self.log_path = self._models_path / "logs" / self.name

	def load_model(self, path, env=None, create=False):
		if env is None:
			env = self.vec_env
		model_kwargs = (
			dict(
				device="cpu",
				env=env,
				n_steps=4096,  # per environment, per rollout
				learning_rate=LinearSchedule(start=1e-3, end=1e-6, end_fraction=1),
				use_sde=True,
				sde_sample_freq=4,
				policy_kwargs=dict(net_arch=[48, 48]),
				clip_range_vf=1.0,
			)
			| self.model_kwargs
		)

		if not isinstance(path, Path) and path is not None:
			path = Path(path)

		if path is not None and path.is_file():
			self.model = PPO.load(path, **model_kwargs)
		elif create:
			self.model = PPO("MlpPolicy", **model_kwargs)
		else:
			return False
		return True

	def get_model(self):
		self.model = None
		_stage = self.stage

		for s in [self.stage, self.stage - 1]:
			self.stage = s
			self.update_name()

			if self.load_model(self.best_model_path):
				print(f"loaded best_model.zip stage {s}")
				break

			if self.load_model(self.model_path):
				print(f"loading model.zip stage {s}")
				break

			# if self.best_model_path.is_file():
			# 	self.model = PPO.load(self.best_model_path, **model_kwargs)
			# 	break
			# elif self.model_path.is_file():
			# 	print(f"loading model.zip stage {s}")
			# 	self.model = PPO.load(self.model_path, **model_kwargs)
			# 	break
		self.stage = _stage
		self.update_name()

		if self.model is None:
			self.load_model(None, self.vec_env, True)
			print("created new model")
			# self.model = PPO("MlpPolicy", **model_kwargs)

	def save(self):
		self.model.save(self.model_path)

	def branch(self, new_kwargs=dict()):
		self.stage += 0.1
		self.update_name()
		self.update_kwargs(self.kwargs | new_kwargs)  # recreates environments
		print("Model branched to:", self.name)
		self.update_config()

	def graduate(self):  # next step in curriculum
		self.stage = math.floor(self.stage) + 1
		self.update_name()
		print("Model graduated to next stage:", self.name)
		self.update_config()  # recreates envs

	def setup_envs(self):
		print(f"Setting up {self.num_cpu} training environments (CPU)")
		self.vec_env = make_vec_env(
			gym_env.BalanceEnv,
			n_envs=self.num_cpu,
			# monitor_dir=str(self.runs_path),
			seed=42,
			env_kwargs=self.kwargs,
			vec_env_cls=SubprocVecEnv,
		)

		self._eval_env = gym_env.BalanceEnv(**self.kwargs)
		self._eval_env.render_mode = "human"
		self.eval_env = Monitor(env=self._eval_env)

		self.eval_callback = EvalCallback(
			self.eval_env,
			best_model_save_path=self.runs_path,
			deterministic=True,
			eval_freq=5000,
			# render=True,
			callback_after_eval=StopTrainingOnNoModelImprovement(
				max_no_improvement_evals=10, min_evals=100, verbose=1
			),
		)

	def update_kwargs(self, kwargs):
		if self.vec_env is not None:
			self.vec_env.close()
			self.eval_env.close()
		self.kwargs = kwargs
		self.setup_envs()
		if self.model is not None:
			self.model.env = self.vec_env

	class SaveProgress(BaseCallback):
		def __init__(self, path):
			self._save_path = path
			super().__init__()

		def _on_step(self):
			_file = self._save_path / f"progress-{self.n_calls}.zip"
			print("saving model in-progress snapshot", _file)
			self.model.save(_file)
			return True

	def train(self, steps):
		print("Training", self.name, steps, "steps")
		self.model.learn(
			total_timesteps=steps,
			progress_bar=True,
			callback=[
				self.eval_callback,
				EveryNTimesteps(
					10000 * self.num_cpu, self.SaveProgress(self.runs_path)
				),
			],
		)
		self.save()

	def eval(self, eps=3):
		print("Evaluating", self.name)
		self._eval_env.debug = True
		evaluate_policy(
			self.model,
			self.eval_env,
			n_eval_episodes=eps,
			deterministic=True,
			render=True,
		)
		self._eval_env.plot()
		# self._eval_env.close()


# def setup_env(num_cpu=1, kwargs=dict(), render=False):
# 	vec_env = make_vec_env(
# 		gym_env.BalanceEnv,
# 		n_envs=num_cpu,
# 		seed=42,
# 		env_kwargs=kwargs,
# 		vec_env_cls=SubprocVecEnv,
# 	)

# 	eval_env = gym_env.BalanceEnv(**kwargs)
# 	eval_env.render_mode = "human"
# 	_eval = EvalCallback(
# 		eval_env,
# 		best_model_save_path=f"./runs/{name}",
# 		deterministic=True,
# 		# eval_freq=10_000,
# 		render=False,
# 	)

# 	if render:
# 		return render_env.VecRender(vec_env, 800, 640)
# 	return vec_env


# def train(name: str, model: BaseAlgorithm, steps):
# 	env = model.get_env()
# 	_eval = EvalCallback(
# 		env,
# 		best_model_save_path=f"./runs/{name}",
# 		deterministic=True,
# 		# eval_freq=10_000,
# 		render=False,
# 	)
