import numpy as np
class BanditInstance:

	def __init__(self, means):
		self.means = means

	def pull(self, arm):
		return int(np.random.rand() < self.means[arm])

