import os
import atexit
import time

from oscpy.server import OSCThreadServer, ServerClass

from training import TrainingSetup

osc = OSCThreadServer()
port = int(os.environ.get("OSC_PORT", 1234))
osc.listen(address="0.0.0.0", port=port, default=True)


class InferServer:
	def __init__(self, t: TrainingSetup, acts):
		self.model = t.load_model(None)
		self.osc = osc
		print(f"osc inference server listening on port {port}")

		self.acts = acts

		self.deviceIP = "192.168.1.143"

		self.act(0.0, 0.0)

		self.f = open("robot.log.new", "w")

		@atexit.register
		def end():
			self.act(0.0, 0.0)
			self.f.close()

		self.osc.bind(b"/obs", self.infer)
		self.osc.join_server()

	def act(self, *_act: list[float]):
		print("act", _act)
		if self.acts:
			self.osc.send_message(b"/act", _act, self.deviceIP, 1212)

	def infer(self, *obs: list[float]):
		obs = list(obs)
		self.deviceIP = self.osc.get_sender()[1]
		print("obs", obs)
		self.f.write(",".join(map(lambda x: str(x), obs)) + "\n")

		s = time.perf_counter_ns()
		act = self.model.predict(obs)[0]
		print(f"infer took {(time.perf_counter_ns() - s) * 1e-6:4.2f}ms")
		self.act(float(act[0]), float(act[1]))
		osc.send_message(
			b"/motor", [obs[3] / 37.0, obs[4] / 37.0, obs[7], obs[8]], "127.0.0.1", 1235
		)
