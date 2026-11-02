from collections import deque
import time

import matplotlib.pyplot as plt
import matplotlib.animation as anim
from oscpy.server import OSCThreadServer

osc = OSCThreadServer()
sock = osc.listen(address="0.0.0.0", port=1235, default=True)

data_motor = deque([(0, 0, 0, 0)], maxlen=2500)
data_motor_sim = deque([(0, 0, 0, 0)], maxlen=2500)
data_balance = deque([(0, 0, 0, 0)], maxlen=2000)


# @osc.address(b"/motor")
# def motor_data(m1Setpoint, m1Input, m1Output, m2Setpoint, m2Input, m2Output):
# 	print(
# 		f"m1: set {m1Setpoint:4.2f} in {m1Input:4.2f} out {m1Output:4.2f}, m2: set {m2Setpoint:4.2f} in {m2Input:4.2f} out {m2Output:4.2f}"
# 	)
# 	data_motor.append((m1Setpoint, m1Input, m1Output, m2Setpoint, m2Input, m2Output))


last_real = (0, 0, 0, 0)


@osc.address(b"/motor")
def motor_data(m1Enc, m2Enc, m1Act, m2Act):
	global last_real
	print(
		f"m1: set {m1Act:4.2f} enc {m1Enc:4.2f}, m2: set {m2Act:4.2f} enc {m2Act:4.2f}"
	)
	last_real = (m1Enc, m1Act, m2Enc, m2Act)
	# data_motor.append((m1Enc, m1Act, m2Enc, m2Act))


@osc.address(b"/motor_sim")
def motor_data_sim(m1Enc, m2Enc, m1Act, m2Act):
	print(
		f"SIM: m1: set {m1Act:4.2f} enc {m1Enc:4.2f}, m2: set {m2Act:4.2f} enc {m2Enc:4.2f}"
	)
	data_motor_sim.append((m1Enc, m1Act, m2Enc, m2Act))
	data_motor.append(last_real)


@osc.address(b"/pid_motor")
def pid_motor(m1set, m1in, m1out, m2set, m2in, m2out):
	print(
		f"m1: set {m1set:4.2f} enc {m1in:4.2f} out {m1out:4.2f}, m2: set {m2set:4.2f} enc {m2in:4.2f} out {m2out:4.2f}"
	)


@osc.address(b"/balance")
def balance(balInput, balOutput, m1Input, m1Output):
	print(
		f"balance: {balInput:4.2f}, {balOutput:4.2f}, {m1Input:4.2f}, {m1Output:4.2f}"
	)
	data_balance.append((balInput, balOutput, m1Input, m1Output))


fig, ax = plt.subplots()
# ax.set_ylim(-35, 35)
ax.set_ylim(-1.2, 1.2)
ax.set_xlim(0, data_motor.maxlen)
lines = [plt.plot([])[0] for _ in range(4)]


def animate(i):
	for x, line in enumerate(lines):
		if x < 2:
			line.set_data(
				[_ for _ in range(len(data_motor))], [d[x] for d in data_motor]
			)
		else:
			line.set_data(
				[_ for _ in range(len(data_motor_sim))],
				[d[x - 2] for d in data_motor_sim],
			)
		# line.set_data(
		# 	[_ for _ in range(len(data_balance))], [d[x] for d in data_balance]
		# )


animate(0)

ani = anim.FuncAnimation(fig, animate)
plt.show()

osc.join_server()
