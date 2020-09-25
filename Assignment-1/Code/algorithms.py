import numpy as np

class EpsilonGreedy:

	def __init__(self, numberOfBandits, epsilon):
		self.numberOfBandits	= numberOfBandits
		self.epsilon 			= epsilon
		self.T 					= -1
		self.pulls   			= np.zeros(numberOfBandits)
		self.rewards 			= np.zeros(numberOfBandits)
		self.means   			= np.zeros(numberOfBandits)

	def getArm(self):
		self.T = self.T + 1
		if np.random.rand() < self.epsilon:
			return np.random.randint(0, self.numberOfBandits)
		else:
			return np.argmax(self.means)

	def getReward(self, arm, reward):
		self.pulls[arm]   += 1
		self.rewards[arm] += reward
		self.means[arm]    = self.rewards[arm] / self.pulls[arm]


class UCB:

	def __init__(self, numberOfBandits):
		self.numberOfBandits	= numberOfBandits
		self.T 					= -1
		self.pulls   			= np.zeros(numberOfBandits)
		self.rewards 			= np.zeros(numberOfBandits)
		self.means   			= np.zeros(numberOfBandits)

	def getArm(self):
		self.T = self.T + 1
		if self.T < self.numberOfBandits:
			return self.T
		else:
			ucb_t = self.means + np.sqrt(2*np.log(self.T)/self.pulls)
			return np.argmax(ucb_t)

	def getReward(self, arm, reward):
		self.pulls[arm]   += 1
		self.rewards[arm] += reward
		self.means[arm]    = self.rewards[arm] / self.pulls[arm]


class KLUCB:

	def __init__(self, numberOfBandits):
		self.numberOfBandits	=	numberOfBandits
		self.precision			=	1e-3
		self.T 					= 	-1
		self.pulls   			= 	np.zeros(numberOfBandits)
		self.rewards 			= 	np.zeros(numberOfBandits)
		self.means   			= 	np.zeros(numberOfBandits)

	def getArm(self):

		self.T = self.T + 1
		if self.T < self.numberOfBandits:
			return self.T

		def binary_search(cpull, cmean, T):
			lo = cmean
			hi = 1
			while hi-lo > self.precision:
				
				mid 	= (lo + hi)/2
				kl_term = (1-cmean)*np.log((1-cmean)/(1-mid))
				if cmean != 0:
					kl_term += cmean*np.log(cmean/mid)

				if cpull*kl_term > np.log(T):
					hi = mid
				else:
					lo = mid
			return lo
		
		kl_ucb_t = np.vectorize(binary_search)
		return np.argmax(kl_ucb_t(self.pulls, self.means, self.T))


	def getReward(self, arm, reward):
		self.pulls[arm]   += 1
		self.rewards[arm] += reward
		self.means[arm]    = self.rewards[arm] / self.pulls[arm]


class ThompsonSampling:

	def __init__(self, numberOfBandits):
		self.numberOfBandits = numberOfBandits
		self.success 		 = np.ones(numberOfBandits) # store success + 1
		self.failure 		 = np.ones(numberOfBandits) # store failure + 1

	def getArm(self):
		return np.argmax([np.random.beta(self.success[i], self.failure[i]) \
			 for i in range(self.numberOfBandits)])

	def getReward(self, arm, reward):
		if reward:
			self.success[arm] += 1
		else:
			self.failure[arm] += 1

class ThompsonSamplingWithHint:

	def __init__(self, numberOfBandits, means):
		self.numberOfBandits = numberOfBandits
		self.means			 = means
		self.prior			 = np.ones((numberOfBandits, numberOfBandits))/numberOfBandits

	def getArm(self):
		return np.argmax(self.prior[np.argmax(self.means), :])
		# return np.argmax([self.means[np.random.choice(self.numberOfBandits, p = self.prior[:, i])] \
		# 	for i in range(self.numberOfBandits)])

	def getReward(self, arm, reward):
		if reward:
			self.prior[:, arm] *= self.means
			self.prior[:, arm] /= np.sum(self.prior[:, arm])
		else:
			self.prior[:, arm] *= 1 - self.means
			self.prior[:, arm] /= np.sum(self.prior[:, arm])