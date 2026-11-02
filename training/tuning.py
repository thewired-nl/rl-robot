#!/usr/bin/env python3

import os
import sys
from typing import Any
import logging
import collections.abc
import time

# tensorflow LLVM JIT breaks mujoco visualization
# sys.modules["tensorboard"] = None
# sys.modules["tensorflow"] = None

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "24")

import gymnasium
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState
import torch

# import torch.nn as nn
import jax.nn as nn

from pathlib import Path

from sbx import PPO

# from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecCheckNan
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import (
	BaseCallback,
	EvalCallback,
	EveryNTimesteps,
	StopTrainingOnNoModelImprovement,
)

import gym_env
from config import configure


def deep_update(d, u):
	for k, v in u.items():
		if isinstance(v, collections.abc.Mapping):
			d[k] = deep_update(d.get(k, {}), v)
		else:
			d[k] = v
	return d


OVERRIDE_ENV_CFG = dict()

N_VERSIONS = 5

N_TRIALS = 500
N_STARTUP_TRIALS = 10
N_TIMESTEPS = 150_000
N_EVALUATIONS = 5
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)  # every 5000 steps, same as training
N_EVAL_EPISODES = 10

N_ENVS = 32

# optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))


def params_to_model(trial: optuna.Trial, params):
	n_steps = 2 ** params["n_steps_pow"]
	trial.set_user_attr("n_steps", n_steps)

	# optimizer_eps = 10**optimizer_eps_pow
	# trial.set_user_attr("optimizer_eps", optimizer_eps)

	# pytorch
	# activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

	# jax (sbx)
	# activation_fn = getattr(nn, activation_fn)

	return {
		"n_steps": n_steps,  # per environment, per rollout
		"gamma": params["gamma"],
		"learning_rate": params["learning_rate"],
		"policy_kwargs": {
			# "activation_fn": activation_fn,
			"optimizer_kwargs": {"eps": 1e-7},
		},
	}


def sample_ppo_params(trial: optuna.Trial) -> dict[str, Any]:
	"""Sampler for PPO hyperparameters."""
	# From 2**5=32 to 2**13=8192
	n_steps_pow = trial.suggest_int("n_steps_pow", 5, 12)
	gamma = trial.suggest_float("gamma", 0.97, 0.9999)
	learning_rate = trial.suggest_float("learning_rate", 3e-5, 3e-3, log=True)
	# activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
	# optimizer_eps_pow = trial.suggest_int("optimizer_epsilon_pow", -9, -3)

	return params_to_model(
		trial, dict(n_steps_pow=n_steps_pow, gamma=gamma, learning_rate=learning_rate)
	)


# EvalCallback that reports to Optuna, and possibly stops this run
# https://araffin.github.io/post/optuna/
class TrialEvalCallback(EvalCallback):
	"""Callback used for evaluating and reporting a trial."""

	def __init__(
		self,
		eval_env: gymnasium.Env,
		trial: optuna.Trial,
		n_eval_episodes: int = 10,
		eval_freq: int = 10000,
		deterministic: bool = True,
		verbose: int = 0,
	):
		super().__init__(
			eval_env=eval_env,
			n_eval_episodes=n_eval_episodes,
			eval_freq=eval_freq,
			deterministic=deterministic,
			verbose=verbose,
		)
		self.trial = trial
		self.eval_idx = 0
		self.is_pruned = False
		self.score_offset = 0
		self.score = 0

	def _on_step(self) -> bool:
		if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
			super()._on_step()
			self.eval_idx += 1
			self.score = self.score_offset + self.last_mean_reward
			self.trial.report(self.score, self.eval_idx)
			# Prune trial if need.
			if self.trial.should_prune():
				self.is_pruned = True
				return False
		return True


def get_config(version=1):
	cfg = configure("hyperparam", version)
	cfg.update(OVERRIDE_ENV_CFG)
	env_kwargs = {k: cfg[k] for k in cfg.keys() - {"model_kwargs"}}
	env_kwargs["version"] = version
	return env_kwargs, cfg["model_kwargs"]


def make_env(env_cfg, cls, seed):
	return VecCheckNan(
		make_vec_env(
			gym_env.BalanceEnv,
			n_envs=N_ENVS,
			seed=seed,
			env_kwargs=env_cfg,
			vec_env_cls=cls,
		),
		raise_exception=True,
	)


def objective_broad(trial: optuna.Trial) -> float:
	return model_objective(trial, sample_ppo_params(trial))


def create_seed_objective(orig_trials):
	trial_numbers = []
	trial_params = {}
	for t in orig_trials:
		trial_numbers.append(t.number)
		trial_params[t.number] = t.params

	def objective_seed(trial: optuna.Trial) -> float:
		num = trial.suggest_categorical("orig_trial", trial_numbers)
		params = trial_params[num]
		trial.set_user_attr("params", params)

		seed_mult = trial.suggest_int("seed_mult", 0, 6)
		seed = seed_mult * (N_ENVS + 1)
		trial.set_user_attr("seed", seed)

		cur_trials = trial.study.get_trials(
			deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED)
		)
		for t in reversed(cur_trials):
			if trial.params == t.params:
				print(
					f"duplicate trial parameters, returning value from {t.number}: {t.value}"
				)
				return t.value

		return model_objective(trial, params_to_model(trial, params), seed)

	return objective_seed


def model_objective(trial: optuna.Trial, model_kwargs, seed=None) -> float:
	env_cfg, base_model_cfg = get_config()
	vec_env = make_env(env_cfg, SubprocVecEnv, seed)

	model_cfg = base_model_cfg.copy()
	deep_update(model_cfg, model_kwargs)

	# Create the RL model.
	model = PPO("MlpPolicy", env=vec_env, **model_cfg)
	eval_env = None
	score = 0

	nan_encountered = False
	try:
		# model.learn(N_TIMESTEPS, callback=eval_callback)

		for ver in range(1, N_VERSIONS + 1):
			start_time = time.time()
			eval_env = make_env(env_cfg, None, seed)
			eval_callback = TrialEvalCallback(
				eval_env,
				trial,
				n_eval_episodes=N_EVAL_EPISODES,
				eval_freq=max(EVAL_FREQ // N_ENVS, 1),
				deterministic=True,
			)

			eval_callback.eval_idx = ver * 100
			eval_callback.score_offset = score

			model.learn(N_TIMESTEPS, callback=eval_callback)
			trial.set_user_attr(
				f"version-{ver}.0",
				dict(
					training_time=time.time() - start_time,
					mean_reward=eval_callback.last_mean_reward,
				),
			)

			score = eval_callback.score

			model.env.close()
			eval_env.close()

			if trial.should_prune():
				break

			if ver != N_VERSIONS:
				# prepare cfg for next version
				env_cfg, _ = get_config(ver + 1)
				model.env = make_env(env_cfg, SubprocVecEnv, seed)
	except AssertionError as e:
		# Sometimes, random hyperparams can generate NaN.
		print(e)
		nan_encountered = True
	finally:
		# Free memory.
		model.env.close()
		if eval_env is not None:
			eval_env.close()

	# Tell the optimizer that the trial failed.
	if nan_encountered:
		return float("nan")

	if eval_callback.is_pruned:
		raise optuna.exceptions.TrialPruned()

	return eval_callback.score


if __name__ == "__main__":
	# Set pytorch num threads to 1 for faster training.
	torch.set_num_threads(1)

	sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS, multivariate=True)
	# Do not prune before 1/8 of the max budget is used.
	pruner = MedianPruner(
		n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 8
	)

	storage = optuna.storages.JournalStorage(
		optuna.storages.journal.JournalFileBackend("./optuna.log")
	)

	study_broad = optuna.create_study(
		study_name="fixed",
		sampler=sampler,
		pruner=pruner,
		direction="maximize",
		storage=storage,
		load_if_exists=True,
	)

	# # evaluate best trials on different seeds
	# best_trials = sorted(
	# 	study_broad.get_trials(deepcopy=False),
	# 	key=lambda t: t.values[0] if t.values is not None else -1,
	# 	reverse=True,
	# )[:10]

	# study_seed = optuna.create_study(
	# 	study_name="curric-progressive-seed",
	# 	sampler=sampler,
	# 	pruner=pruner,
	# 	direction="maximize",
	# 	storage=storage,
	# 	load_if_exists=True,
	# )

	# objective_seed = create_seed_objective(best_trials)

	# for t in study_seed.get_trials(deepcopy=False):
	# 	if t.state is TrialState.FAIL:
	# 		study_seed.enqueue_trial(
	# 			t.params, user_attrs={"retry": t.user_attrs.get("retry", 0) + 1}
	# 		)

	# study_seed.optimize(objective_seed, n_trials=100, timeout=None, gc_after_trial=True)

	try:
		study_broad.optimize(objective_broad, n_trials=N_TRIALS, timeout=None)
	except KeyboardInterrupt:
		pass

	print(f"Number of finished trials: {len(study_broad.trials)}")

	print("Best trial:")
	trial = study_broad.best_trial

	print("  Value: ", trial.value)

	print("  Params: ")
	for key, value in trial.params.items():
		print(f"    {key}: {value}")

	print("  User attrs:")
	for key, value in trial.user_attrs.items():
		print(f"    {key}: {value}")
