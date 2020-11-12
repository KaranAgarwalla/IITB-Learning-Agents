import numpy as np

class gridWorld:

	def __init__(self, gridParams):
		self.windStrength 		= gridParams['windStrength'].copy()
		self.numMoves			= gridParams['numMoves']
		self.stochastic			= gridParams['Stochastic']
		self.numRows			= gridParams['numRows']
		self.numCols 			= gridParams['numCols']
		self.startState			= gridParams['startState'].copy()
		self.endState 			= gridParams['endState'].copy()
		self.currState 			= self.startState.copy()
		self.actionDic			= {}
		# See the meanings below
		self.actionDic.update({
			0: (-1, 0),
			1: (0, 1),
			2: (1, 0),
			3: (0, -1),
			4: (-1, 1),
			5: (1, 1),
			6: (1, -1),
			7: (-1, -1),
			8: (0, 0),
			})

	def nextMove(self, action):
		'''
			Returns nextState, reward
			Actions list:
				0: North
				1: East
				2: South
				3: West
				4: North East
				5: South East
				6: South West
				7: North West
				8: Stay Put
		'''

		if action >= self.numMoves:
			raise RuntimeError('Invalid Move')

		reward = -1
		windSpeed = 0

		if self.stochastic and self.windStrength[self.currState[1]] > 0:
			windSpeed = self.windStrength[self.currState[1]] + np.random.randint(-1, 2)
		else:
			windSpeed = self.windStrength[self.currState[1]]

		self.currState[0]  = self.currState[0] + self.actionDic[action][0] - windSpeed
		self.currState[1]  = self.currState[1] + self.actionDic[action][1]

		self.currState[0]  = min(max(self.currState[0], 0), self.numRows - 1)
		self.currState[1]  = min(max(self.currState[1], 0), self.numCols - 1)

		if np.array_equal(self.currState, self.endState):
			self.currState 	= self.startState.copy()
			return self.endState.copy(), reward

		return self.currState.copy(), reward