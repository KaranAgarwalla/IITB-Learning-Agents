from gridWorld import gridWorld
from model import Model
import numpy as np 
import argparse
import matplotlib.pyplot as plt

def getAction(Q, epsilon, currState):
	if np.random.rand() < epsilon:
		return np.random.randint(0, Q.shape[2])
	else:
		return np.argmax(Q[currState[0], currState[1], :])


def sarsaAgent(epsilon, learning_rate, episodes, grid, seed, flag):
	np.random.seed(seed)
	time_steps = np.zeros(episodes+1)
	Q = np.zeros((grid.numRows, grid.numCols, grid.numMoves))
	timer = 0

	for i in range(1, episodes+1):
		
		currState = grid.startState.copy()
		
		currAction = getAction(Q, epsilon, currState)
		timer += 1
		
		nextState, reward = grid.nextMove(currAction)

		while not np.array_equal(nextState, grid.endState):
			
			nextAction = getAction(Q, epsilon, nextState)
			timer += 1

			target = reward + Q[nextState[0]][nextState[1]][nextAction]
			Q[currState[0]][currState[1]][currAction] *= 1 - learning_rate
			Q[currState[0]][currState[1]][currAction] += learning_rate*target
			
			currState = nextState.copy()
			currAction = nextAction
			nextState, reward = grid.nextMove(currAction)

		Q[currState[0]][currState[1]][currAction] *= 1 - learning_rate
		Q[currState[0]][currState[1]][currAction] += learning_rate*reward

		time_steps[i] = timer

	if flag:
		return time_steps
	else:
		return Q

def qLearningAgent(epsilon, learning_rate, episodes, grid, seed, flag):
	np.random.seed(seed)
	time_steps = np.zeros(episodes+1)
	Q = np.zeros((grid.numRows, grid.numCols, grid.numMoves))
	timer = 0

	for i in range(1, episodes+1):
		
		currState = grid.startState.copy()

		while not np.array_equal(currState, grid.endState):
			
			currAction = getAction(Q, epsilon, currState)
			nextState, reward = grid.nextMove(currAction)
			timer += 1

			target = reward + np.amax(Q[nextState[0], nextState[1], :])
			Q[currState[0]][currState[1]][currAction] *= 1 - learning_rate
			Q[currState[0]][currState[1]][currAction] += learning_rate*target
			
			currState = nextState.copy()

		time_steps[i] = timer

	if flag:
		return time_steps
	else:
		return Q

def expectedSarsaAgent(epsilon, learning_rate, episodes, grid, seed, flag):
	np.random.seed(seed)
	time_steps = np.zeros(episodes+1)
	Q = np.zeros((grid.numRows, grid.numCols, grid.numMoves))
	timer = 0

	for i in range(1, episodes+1):
		
		currState = grid.startState.copy()

		while not np.array_equal(currState, grid.endState):
			
			currAction = getAction(Q, epsilon, currState)
			nextState, reward = grid.nextMove(currAction)
			timer += 1

			policy = np.ones(grid.numMoves)*epsilon/grid.numMoves
			policy[np.argmax(Q[nextState[0], nextState[1], :])] += 1 - epsilon
			target = reward + np.sum(Q[nextState[0], nextState[1], :]*policy)

			Q[currState[0]][currState[1]][currAction] *= 1 - learning_rate
			Q[currState[0]][currState[1]][currAction] += learning_rate*target
			
			currState = nextState.copy()

		time_steps[i] = timer
	if flag:
		return time_steps
	else:
		return Q

def dynaQAgent(epsilon, learning_rate, episodes, grid, seed, planningSteps, flag):
	np.random.seed(seed)
	time_steps = np.zeros(episodes+1)
	# Q Learning Parameters
	Q 	= np.zeros((grid.numRows, grid.numCols, grid.numMoves))
	# The Model
	model = Model(grid.numRows, grid.numCols, grid.numMoves)
	timer = 0

	for i in range(1, episodes+1):
		
		currState = grid.startState.copy()

		while not np.array_equal(currState, grid.endState):
			
			currAction = getAction(Q, epsilon, currState)
			nextState, reward = grid.nextMove(currAction)
			timer += 1

			target = reward + np.amax(Q[nextState[0], nextState[1], :])
			Q[currState[0]][currState[1]][currAction] *= 1 - learning_rate
			Q[currState[0]][currState[1]][currAction] += learning_rate*target
			
			model.updateModel(currState, currAction, nextState, reward)
			currState = nextState.copy()

			for steps in range(planningSteps):
				# Update State, Action, Next State, Reward
				us, ua, uns, ur = model.obtainTransition()
				ut = ur + np.amax(Q[uns[0], uns[1], :])
				Q[us[0]][us[1]][ua] *= 1 - learning_rate
				Q[us[0]][us[1]][ua] += learning_rate * ut

		time_steps[i] = timer

	if flag:
		return time_steps
	else:
		return Q

def showPath(Q, grid, title):
	gridDisplay = np.zeros((grid.numRows, grid.numCols))
	currState = grid.startState.copy()
	gridDisplay[currState[0]][currState[1]] = 1
	
	while not np.array_equal(currState, grid.endState):
		
		currAction = getAction(Q, 0, currState)
		nextState, reward = grid.nextMove(currAction)		
		currState = nextState.copy()
		if gridDisplay[currState[0]][currState[1]] == 2 and not grid.Stochastic:
			raise RuntimeError('In A Cycle for Optimal Path')
		gridDisplay[currState[0]][currState[1]] = 2
	
	gridDisplay[currState[0]][currState[1]] = 3

	plt.figure(figsize=(grid.numRows, grid.numCols))
	plt.imshow(gridDisplay, cmap=plt.cm.CMRmap, interpolation='nearest', aspect='equal')
	ax = plt.gca()
	ax.set_xticks(np.arange(0, grid.numCols, 1))
	ax.set_yticks(np.arange(0, grid.numRows, 1))
	ax.set_xticklabels(np.arange(0, grid.numCols, 1))
	ax.set_yticklabels(np.arange(0, grid.numRows, 1))
	ax.set_xticks(np.arange(-.5, grid.numCols, 1), minor=True)
	ax.set_yticks(np.arange(-.5, grid.numRows, 1), minor=True)
	ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
	plt.savefig(f'{title}.png')

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	### Grid Input
	parser.add_argument("--numRows", type = int, default = 7, help="Number of rows in grid")
	parser.add_argument("--numCols", type = int, default = 10, help="Number of columns in grid")
	parser.add_argument("--windStrength", type = str, default = "0 0 0 1 1 1 2 2 1 0", help="Enter wind strength as string")
	parser.add_argument("--Stochastic", default = False, action='store_true', help="Stochastic Grid")
	parser.add_argument("--startState", type = str, default = "3 0", help="Enter start state as string")
	parser.add_argument("--endState", type = str, default = "3 7", help="Enter end state as string")
	parser.add_argument("--numMoves", type = int, default = 4, choices = [4, 8, 9], help="Baseline, King's Moves, 9 Moves")

	### Algorithm related parameters
	parser.add_argument("--algorithm", default='Sarsa', choices = ['Sarsa', 'QLearning', 'ExpectedSarsa', 'DynaQ', 'all'])
	parser.add_argument("--epsilon", type = float, default = 0.1, help="epsilon")
	parser.add_argument("--learning_rate", type = float, default = 0.5, help="Learning Rate")
	parser.add_argument("--episodes", type = int, default = 170, help="Number of episodes")
	parser.add_argument("--numSeeds", type = int, default = 50, help="Number of seeds to average across")
	parser.add_argument("--showPath", default= False, action='store_true', help="Show Greedy Path")
	parser.add_argument("--taskID", type = str, default = "", help="Task Number")
	parser.add_argument("--title", type = str, default = "Episodes vs TimeSteps", help="Title of the plot")
	parser.add_argument("--planningSteps", type = int, default = 0, help="Number of Planning Steps in DynaQ Algorithm")
	args = parser.parse_args()

	gridParams  	= {}
	gridParams['numRows']			= args.numRows
	gridParams['numCols']			= args.numCols
	gridParams['windStrength']		= np.array(args.windStrength.split(' '), dtype = np.int64)
	gridParams['Stochastic']      	= args.Stochastic
	gridParams['startState']      	= np.array(args.startState.split(' '), dtype = np.int64)
	gridParams['endState']        	= np.array(args.endState.split(' '), dtype = np.int64)
	gridParams['numMoves']			= args.numMoves

	epsilon 		= args.epsilon
	learning_rate 	= args.learning_rate
	episodes		= args.episodes
	numSeeds		= args.numSeeds

	algorithm_lst = []
	if args.algorithm == 'all':
		algorithm_lst.append('Sarsa')
		algorithm_lst.append('QLearning')
		algorithm_lst.append('ExpectedSarsa')
		algorithm_lst.append('DynaQ')
	else:
		algorithm_lst.append(args.algorithm)

	grid = gridWorld(gridParams)
	time_steps_sarsa = None
	time_steps_qLearning = None
	time_steps_expectedSarsa = None
	time_steps_dynaQ = None

	if 'Sarsa' in algorithm_lst:
		time_steps_sarsa 		= np.array([sarsaAgent(epsilon, learning_rate, episodes, grid, seed, 1) for seed in range(numSeeds)])
		time_steps_sarsa 		= np.mean(time_steps_sarsa, axis = 0)
		if args.showPath:
			Q 	= np.array([sarsaAgent(epsilon, learning_rate, episodes, grid, seed, 0) for seed in range(numSeeds)])
			Q 	= np.mean(Q, axis = 0)
			showPath(Q, grid, f'Task {args.taskID} Sarsa(0) Agent Path')
	
	if 'QLearning' in algorithm_lst:
		time_steps_qLearning 	= np.array([qLearningAgent(epsilon, learning_rate, episodes, grid, seed, 1) for seed in range(numSeeds)])
		time_steps_qLearning 	= np.mean(time_steps_qLearning, axis = 0)
		if args.showPath:
			Q 	= np.array([qLearningAgent(epsilon, learning_rate, episodes, grid, seed, 0) for seed in range(numSeeds)])
			Q 	= np.mean(Q, axis = 0)
			showPath(Q, grid, f'Task {args.taskID} QLearning Agent Path')
	
	if 'ExpectedSarsa' in algorithm_lst:
		time_steps_expectedSarsa= np.array([expectedSarsaAgent(epsilon, learning_rate, episodes, grid, seed, 1) for seed in range(numSeeds)])
		time_steps_expectedSarsa= np.mean(time_steps_expectedSarsa, axis = 0)
		if args.showPath:
			Q 	= np.array([expectedSarsaAgent(epsilon, learning_rate, episodes, grid, seed, 0) for seed in range(numSeeds)])
			Q 	= np.mean(Q, axis = 0)
			showPath(Q, grid, f'Task {args.taskID} Expected Sarsa Agent Path')
	
	if 'DynaQ' in algorithm_lst:
		time_steps_dynaQ 		= np.array([dynaQAgent(epsilon, learning_rate, episodes, grid, seed, args.planningSteps, 1) for seed in range(numSeeds)])
		time_steps_dynaQ 		= np.mean(time_steps_dynaQ, axis = 0)
		if args.showPath:
			Q 	= np.array([dynaQAgent(epsilon, learning_rate, episodes, grid, seed, args.planningSteps, 0) for seed in range(numSeeds)])
			Q 	= np.mean(Q, axis = 0)
			showPath(Q, grid, f'Task {args.taskID} dynaQ Agent Path')
	
	episodes_y = np.arange(episodes+1, dtype = np.int64)
	plt.figure()
	if 'Sarsa' in algorithm_lst:
		plt.plot(time_steps_sarsa, episodes_y, label = "Sarsa")
	if 'QLearning' in algorithm_lst:
		plt.plot(time_steps_qLearning, episodes_y, label = "Q Learning")
	if 'ExpectedSarsa' in algorithm_lst:
		plt.plot(time_steps_expectedSarsa, episodes_y, label = "Expected Sarsa")
	if 'DynaQ' in algorithm_lst:
		plt.plot(time_steps_dynaQ, episodes_y, label = "Dyna Q")

	if len(algorithm_lst) > 1:
		plt.legend()

	plt.title(f'Task:{args.taskID} {args.title}')
	plt.xlabel("Time steps")
	plt.ylabel("Episodes")
	plt.savefig(f'Task {args.taskID}')
	
