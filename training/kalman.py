# Python rewrite of:

# KalmanFilter.cpp - Class file for the Kalman Filter
# Version: 1.0.0
# (c) 2014 Korneliusz Jarzebski
# www.jarzebski.pl
# Licensed GPL-3.0

import numpy as np


class KalmanFilter:
	def __init__(self, angle, bias, measure):
		self.Q_angle = angle
		self.Q_bias = bias
		self.R_measure = measure

		self.K_angle = 0.0
		self.K_bias = 0.0
		self.K_rate = 0.0

		self.P = np.zeros((2, 2))
		self.K = np.zeros((2))

		self.kt = 0.0

	def update(self, now_s, newValue, newRate):
		dt = now_s - self.kt

		self.K_rate = newRate - self.K_bias
		self.K_angle += dt * self.K_rate

		self.P[0][0] += dt * (self.P[1][1] + self.P[0][1]) + self.Q_angle * dt
		self.P[0][1] -= dt * self.P[1][1]
		self.P[1][0] -= dt * self.P[1][1]
		self.P[1][1] += self.Q_bias * dt

		S = self.P[0][0] + self.R_measure

		self.K[0] = self.P[0][0] / S
		self.K[1] = self.P[1][0] / S

		y = newValue - self.K_angle

		self.K_angle += self.K[0] * y
		self.K_bias += self.K[1] * y

		self.P[0][0] -= self.K[0] * self.P[0][0]
		self.P[0][1] -= self.K[0] * self.P[0][1]
		self.P[1][0] -= self.K[1] * self.P[0][0]
		self.P[1][1] -= self.K[1] * self.P[0][1]

		self.kt = now_s

		return self.K_angle
