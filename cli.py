#!/usr/bin/env python3

import os
import json
import argparse
import sys
from pathlib import Path

from config import configure
from dataclasses import asdict

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "24"


def save_model_info():
	try:
		j = json.dumps(model_info, indent=2)
		with open("models.json", "w") as f:
			f.write(j)
	except Exception as e:
		print(e)


try:
	with open("models.json", "r") as f:
		model_info = json.load(f)
except Exception as e:
	print(e)
	model_info = dict(models=dict())
	save_model_info()


def _ver(ver):
	return f"{ver:.1f}"


def namever(name, ver):
	return f"{name}-{_ver(ver)}"


class ModelPaths:
	_models_dir = Path("models")

	def __init__(self, namever):
		self.model_dir = self._models_dir / namever
		self.best = self.model_dir / "best_model.zip"
		self.tensorboard = self._models_dir / "logs" / namever

	def progress(self, step) -> Path:
		return self.model_dir / f"progress-{step}.zip"


def resolve_spec(namever: str, spec="best", check=True):
	mp = ModelPaths(namever)
	if spec == "best":
		path = mp.best
	else:  # newest
		try:
			path = max(
				mp.model_dir.glob(mp.progress("*").relative_to(mp.model_dir)),
				key=lambda x: float(x.stem.split("-")[1]),
			)
		except Exception:
			path = mp.best

	if check and not path.is_file():
		raise Exception(f"Expected `{spec}` model zip `{path}` to exist")
	return path


def model_config(name, version, with_kwargs=False, serialize=False):
	cfg = configure(name, version)
	if with_kwargs:
		cfg.update(
			model_info["models"][name]["configs"][_ver(version)].get(
				"env_kwargs", dict()
			)
		)

	if not serialize:
		return cfg

	def diffdict(_class):
		defaults = _class.__class__()
		return asdict(
			_class,
			dict_factory=lambda x: {k: v for (k, v) in x if v != getattr(defaults, k)},
		)

	return dict(
		coef=diffdict(cfg["coef"]),
		dr=diffdict(cfg["dr"]),
		# model_kwargs=cfg["model_kwargs"], # not serializable
	)


def match_model(specifier, check=True, _spec=None):
	if specifier is None:
		specifier = model_info.get("active", None)
	if specifier is None:
		sys.exit("No model specifier provided, and none actively training")
	# valid specifiers:
	# `name` / `name:best` (best_model, latest version)
	# `name:latest` (latest progress, latest version)
	# `name-2.1` (specific version)
	# `name-2.1:latest` (latest progress, specific version)
	name, *spec = specifier.split(":", 1)
	spec = spec[0] if spec else "best"
	if _spec:
		spec = _spec
	nameparts = name.split("-")
	version = None
	if len(nameparts) >= 2:
		try:
			version = float(nameparts[-1])
			name = "-".join(nameparts[:-1])
		except ValueError:
			name = "-".join(nameparts)
			pass
	# print(
	# 	f"looking for `{name}`{f' version `{version:.1f}`' if version is not None else ''} spec `{spec}`"
	# )

	if name not in model_info["models"]:
		sys.exit(f"No models named `{name}`")

	model = model_info["models"][name]

	if version is None:
		version = max(model["versions"])
	elif version not in model["versions"]:
		sys.exit(
			f"Model `{name}` has no version `{_ver(version)}`. Available versions: [ {', '.join(model['versions'])} ]"
		)

	spec_path = resolve_spec(namever(name, version), spec, check=check)
	return (name, version, spec_path)


def add_model(name, version=1, _input=None):
	model = model_info["models"][name]

	if version in model["versions"]:
		raise Exception(f"Model `{name}` already has version `{_ver(version)}`")

	model["versions"].append(version)

	cfg = model_config(name, version, serialize=True)

	if _input is not None:
		if isinstance(_input, str):
			iname, iver, ipath = match_model(_input)
			cfg["input"] = namever(iname, iver)
			cfg["input_path"] = str(ipath)
		else:
			cfg.update(_input)

	model["configs"][_ver(version)] = cfg
	model_info["active"] = namever(name, version)
	save_model_info()


def create(args):
	if args.name not in model_info["models"]:
		model_info["models"][args.name] = dict(versions=[], configs={})

	add_model(args.name, args.version, args.input)


def _next(args):
	iname, iver, ipath = match_model(args.name)
	add_model(
		iname, iver + args.inc, dict(input=namever(iname, iver), input_path=str(ipath))
	)


def train(args):
	from training import TrainingSetup

	name, ver, path = match_model(args.name, check=False)
	_namever = namever(name, ver)
	if model_info["active"] is not _namever:
		model_info["active"] = _namever
		save_model_info()
	print("training", _namever, "for", args.count, "steps")
	cfg = model_config(name, ver, with_kwargs=True)

	mp = ModelPaths(_namever)
	try:
		path = resolve_spec(_namever, "latest")
	except Exception:
		_cfg = model_info["models"][name]["configs"][_ver(ver)]
		path = Path(_cfg.get("input_path", path))

	t = TrainingSetup(path, cfg, mp)
	t.train(args.count)


def _eval(args):
	from training import TrainingSetup

	name, ver, path = match_model(args.name, True, args.spec)
	mp = ModelPaths(namever(name, ver))
	cfg = model_config(name, ver, with_kwargs=True)
	t = TrainingSetup(path, cfg, mp)
	t.eval(args.count)


def select(args):
	name, ver, path = match_model(args.spec, check=False)
	print(path)
	model_info["active"] = namever(name, ver)
	save_model_info()


def change(args):
	name, ver, path = match_model(args.name, check=False)
	env_kwargs = model_info["models"][name]["configs"][_ver(ver)].get(
		"env_kwargs", dict()
	)
	env_kwargs["model_xml"] = args.xml
	model_info["models"][name]["configs"][_ver(ver)]["env_kwargs"] = env_kwargs

	save_model_info()


def rm(args):
	del model_info["models"][args.name]
	if model_info["active"] == args.name:
		del model_info["active"]
	save_model_info()


def check_env(args):
	from gym_env import BalanceEnv
	from stable_baselines3.common.env_checker import check_env

	name, ver, path = match_model(args.name, check=False)
	cfg = model_config(name, ver, with_kwargs=True)
	del cfg["model_kwargs"]
	env = BalanceEnv(**cfg)
	check_env(env)


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	sp = ap.add_subparsers(required=True, help="subcommand help")

	ap_create = sp.add_parser("create")
	ap_create.add_argument("name")
	ap_create.add_argument("--version", "--ver", default=1, type=float)
	ap_create.add_argument("--input", default=None)
	ap_create.set_defaults(func=create)

	ap_next = sp.add_parser("next")
	ap_next.add_argument("--name")
	ap_next.add_argument("--inc", default=1, type=float)
	ap_next.set_defaults(func=_next)

	ap_train = sp.add_parser("train")
	ap_train.add_argument("name", nargs="?")
	ap_train.add_argument("--count", "-c", default=16_000_000, type=int)
	ap_train.set_defaults(func=train)

	ap_eval = sp.add_parser("eval")
	ap_eval.add_argument("name", nargs="?")
	ap_eval.add_argument("--spec", "-s", nargs="?", default="latest")
	ap_eval.add_argument("--count", "-c", default=1, type=int)
	ap_eval.set_defaults(func=_eval)

	ap_select = sp.add_parser("select")
	ap_select.add_argument("spec", nargs="?")
	ap_select.set_defaults(func=select)

	ap_change = sp.add_parser("change")
	ap_change.add_argument("name", nargs="?")
	ap_change.add_argument("--xml")
	ap_change.set_defaults(func=change)

	ap_rm = sp.add_parser("rm")
	ap_rm.add_argument("name")
	ap_rm.set_defaults(func=rm)

	ap_check_env = sp.add_parser("check_env")
	ap_check_env.add_argument("name", nargs="?")
	ap_check_env.set_defaults(func=check_env)

	args = ap.parse_args()
	args.func(args)
