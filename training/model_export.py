import torch as th
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.evaluation import evaluate_policy
import onnxruntime

from esp_ppq.api import espdl_quantize_onnx
from esp_ppq.executor.torch import TorchExecutor

# adapted from https://stable-baselines3.readthedocs.io/en/master/guide/export.html#sbx-sb3-jax-export


class TorchPolicy(th.nn.Module):
	def __init__(self, obs_dim: int, hidden_dim: int, act_dim: int):
		super().__init__()
		self.net = th.nn.Sequential(
			th.nn.Linear(obs_dim, hidden_dim),
			th.nn.Tanh(),
			th.nn.Linear(hidden_dim, hidden_dim),
			th.nn.Tanh(),
			th.nn.Linear(hidden_dim, act_dim),
		)

	def forward(self, x: th.Tensor) -> th.Tensor:
		return self.net(x)


class EnvObserver(VecEnvWrapper):
	def __init__(self, venv: VecEnv):
		self.actions = []
		self.obs = []
		super().__init__(venv=venv)

	def obs_(self, res, reset=False):
		if reset:
			for i in range(self.num_envs):
				self.obs.append(res[0])
		else:
			for o in res[0]:
				self.obs.append(o)
		return res

	def reset(self):
		return self.obs_(self.venv.reset(), True)

	def step_async(self, actions: np.ndarray):
		for a in actions:
			self.actions.append(a)
		super().step_async(actions)

	def step_wait(self):
		return self.obs_(self.venv.step_wait())

	def close(self):
		self.venv.close()
		super().close()

	def collect(self):
		# trim additional final reset observations
		o = np.stack(self.obs[: len(self.obs) - self.num_envs])
		a = np.stack(self.actions)
		return o, a


class ONNXPolicy:
	def __init__(self, path):
		self.session = onnxruntime.InferenceSession(path)
		print("inputs:", [x.name for x in self.session.get_inputs()])
		print("outputs:", [x.name for x in self.session.get_outputs()])

	def predict(
		self, observation, state=None, episode_start=None, deterministic: bool = True
	):
		# print("predict")
		actions = []
		for o in observation:
			# print("  obs:", o)
			act = self.session.run(["act"], {"obs": [o]})
			actions.append(act[0][0])
			# print("  act:", act)
			# print()
		# print("  action:", action)
		return (actions, None)


class ModelExport:
	from training import TrainingSetup

	def __init__(self, t: TrainingSetup, force=False):
		self.t = t
		self.force = force
		self.model = self.t.load_model(None)

	def convert(self):
		print("converting to PyTorch model")
		model = self.model
		params = model.policy.actor_state.params["params"]
		# For debug:
		print("=== SBX params ===")
		for key, value in params.items():
			if isinstance(value, dict):
				for name, val in value.items():
					print(f"{key}.{name}: {val.shape}", end=" ")
			else:
				print(f"{key}: {value.shape}", end=" ")
		print("\n" + "=" * 20 + "\n")

		obs_dim = model.observation_space.shape
		act_dim = model.action_space.shape

		# Number of units in the hidden layers (assume a network architecture like [64, 64])
		hidden_dim = params["Dense_0"]["kernel"].shape[1]

		# map params to torch state_dict keys
		num_layers = len([k for k in params.keys() if k.startswith("Dense_")])
		state_dict = {}
		for i in range(num_layers):
			layer_name = f"Dense_{i}"
			state_dict[f"net.{i * 2}.bias"] = th.from_numpy(
				np.array(params[layer_name]["bias"])
			)
			state_dict[f"net.{i * 2}.weight"] = th.from_numpy(
				np.array(params[layer_name]["kernel"].T)
			)

		torch_policy = TorchPolicy(obs_dim[0], hidden_dim, act_dim[0])
		print("=== Torch params ===")
		print(
			" ".join(
				f"{key}:{tuple(value.shape)}"
				for key, value in torch_policy.named_parameters()
			)
		)
		print("=" * 20 + "\n")

		torch_policy.load_state_dict(state_dict)
		torch_policy.eval()

		torch_export = th.export.export(torch_policy, (th.zeros((1, *obs_dim)),))
		th.export.save(torch_export, "test.pt2")

		self.model_th = torch_policy

		th.onnx.export(
			torch_policy,
			(th.zeros((1, *obs_dim)),),
			str(self.t.mp.onnx),
			opset_version=18,
			input_names=["obs"],
			output_names=["act"],
			dynamo=True,
		)

	# def obs_onnx_sample(self):
	# 	if self.force or not self.t.mp.onnx.is_file():
	# 		self.save_onnx()

	# 	model = ONNXPolicy(self.t.mp.onnx)

	# 	evaluate_policy(model, self.model.env, n_eval_episodes=1, deterministic=True)
	# 	print(f"collected {len(self.model.env.obs)} action/observation pairs")

	def quantized_export(self, target="esp32s3"):
		if self.force or not self.t.mp.onnx.is_file():
			self.convert()

		onnx_model = ONNXPolicy(self.t.mp.onnx)

		obs = []
		acts = []

		with open("robot.log", "r") as f:
			for line in f:
				_obs = list(map(lambda x: float(x), line.split(",")))
				obs.append(_obs)
				acts.append(self.model.predict(_obs)[0])

		# for i in range(64000):
		# 	_obs = self.model.observation_space.sample()
		# 	obs.append(_obs)
		# 	acts.append(self.model.predict(_obs)[0])

		o = np.stack(obs)
		a = np.stack(acts)

		# o, a = self.model.env.collect()

		dataset = th.utils.data.TensorDataset(th.as_tensor(o), th.as_tensor(a))
		dataloader = th.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

		quant_graph = espdl_quantize_onnx(
			# model=self.model_th,
			onnx_import_file=str(self.t.mp.onnx),
			espdl_export_file=str(self.t.mp.espdl),
			calib_dataloader=dataloader,
			calib_steps=32,
			input_shape=[1, self.model.observation_space.shape[0]],
			inputs=[th.from_numpy(self.model.observation_space.sample())],
			target=target,
			# quant_type="w8a8",
			quant_type="w16a16",
			collate_fn=lambda x: x[0].to("cpu"),
			device="cpu",
			error_report=True,
			skip_export=False,
			export_test_values=True,
			verbose=True,
		)

		print()

		criterion = th.nn.MSELoss()
		loss = 0
		for batch_x, batch_y in dataloader:
			# y_pred = self.model_th(batch_x)
			y_pred = th.as_tensor(
				np.clip(
					np.stack(onnx_model.predict(batch_x)[0]),
					self.model.action_space.low,
					self.model.action_space.high,
				)
			)
			# loss_ = criterion(y_pred, batch_y)
			# if loss_ > 1:
			# 	print(batch_x, batch_y, y_pred)
			loss += criterion(y_pred, batch_y)
		loss /= len(dataloader)
		print(f"ONNX model loss: {loss.item():.5f}")

		executor = TorchExecutor(graph=quant_graph, device="cpu")
		loss = 0
		for batch_x, batch_y in dataloader:
			y_pred = th.as_tensor(
				np.clip(
					executor(batch_x)[0],
					self.model.action_space.low,
					self.model.action_space.high,
				)
			)
			loss += criterion(y_pred, batch_y)
		loss /= len(dataloader)
		print(f"Quantized model loss: {loss.item():.5f}")

		print()
		print(f"Exported {self.t.mp.espdl}")
