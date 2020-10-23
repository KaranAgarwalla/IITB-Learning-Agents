import numpy as np 
import argparse
import matplotlib.pyplot as plt
import pulp

def read(file):
	numStates, numActions = 0, 0
	startState, endStates = 0, 0
	transition, reward = 0, 0
	mdptype, gamma = 0, 0

	with open(file) as f:
			for l in f:
				cl = l.strip('\n').split()

				if cl[0] == 'numStates':
					numStates = int(cl[1])
				
				elif cl[0] == 'numActions':
					numActions = int(cl[1])
					transition = np.zeros((numStates, numActions, numStates), dtype=np.float64)
					reward = np.zeros((numStates, numActions, numStates), dtype=np.float64)
				
				elif cl[0] == 'start':
					startState = int(cl[1])
				
				elif cl[0] == 'end':
					endStates = []
					for i in range(1, len(cl)):
						endStates.append(int(cl[i]))
					endStates = np.array(endStates)
				
				elif cl[0] == 'transition':
					s1, ac, s2, r, p = int(cl[1]), int(cl[2]), int(cl[3]), float(cl[4]), float(cl[5])
					transition[s1][ac][s2] = p
					reward[s1][ac][s2] = r
				
				elif cl[0] == 'mdptype':
					mdptype = cl[1]
				
				elif cl[0] == 'discount':
					gamma = float(cl[1])
				
				else:
					print("Aborting: Incorrect Format")
					exit(0)

	return numStates, numActions, startState, endStates, transition, reward, mdptype, gamma

def actionValueFunction(numStates, numActions, transition, reward, gamma, V):
	'''
		returns Q value function for given policy and valueFunction
	'''
	Q = np.sum(transition*reward, axis = 2) + gamma*transition@V
	return Q

def valueFunction(numStates, numActions, transition, reward, gamma, policy):
	'''
		Returns the value function given the policy
	'''

	transition_policy = np.array([transition[i, policy[i], :] for i in range(numStates)])
	reward_policy 	  = np.array([reward[i, policy[i], :] for i in range(numStates)])
	A = np.identity(numStates) - gamma*transition_policy
	B = np.sum(transition_policy*reward_policy, axis = 1)
	return np.linalg.pinv(A)@B

def valueIteration(numStates, numActions, transition, reward, gamma):
	'''
		Returns optimal policy using Value Iteration
	'''
	V = np.zeros(numStates)
	tolerance = 1e-10
	diff = 1
	while diff>tolerance:
		nV = np.amax(actionValueFunction(numStates, numActions, transition, reward, \
			gamma, V), axis = 1)
		diff = np.linalg.norm(V - nV)
		V = nV
	return V


def howardPolicyIteration(numStates, numActions, transition, reward, gamma):
	'''
		Implements Howard Policy Iteration to find optimal value function V
	'''
	policy = np.zeros(numStates, dtype = np.int64)
	V = valueFunction(numStates, numActions, transition, reward, gamma, policy)
	
	while True:
		Q = actionValueFunction(numStates, numActions, transition, reward, gamma, V)
		nPolicy = np.argmax(Q, axis = 1)
		if np.array_equal(policy, nPolicy):
			break
		policy = nPolicy
		V = valueFunction(numStates, numActions, transition, reward, gamma, policy)

	return V, policy


def linearProgrammingSolver(numStates, numActions, transition, reward, gamma):
	'''
		Implements linear programming solution to find optimal value function V
	'''
	prob = pulp.LpProblem("valueFunction", pulp.LpMinimize)
	V = pulp.LpVariable.dicts('valueFunction', range(numStates))

	#Formulate Objective Function
	prob += pulp.lpSum([V[s] for s in range(numStates)])

	#Formulate constraints
	for s in range(numStates):
		for a in range(numActions):
			prob += pulp.lpSum([transition[s, a, nS]*(reward[s, a, nS] + gamma*V[nS]) \
			 for nS in range(numStates)]) <= V[s]

	solver = pulp.PULP_CBC_CMD(msg=False)
	prob.solve(solver)

	valueF = np.zeros(numStates)
	for i in range(numStates):
		valueF[i] = pulp.value(V[i])
	return valueF

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--mdp", help="Path to the mdp file")
	parser.add_argument("--algorithm", choices = ['vi', 'hpi', 'lp'], \
		help="Value Iteration, Howard Policy Iteration, Linear Programming")
	args = parser.parse_args()

	numStates, numActions, startState, endStates, transition, reward, mdptype, gamma = read(args.mdp)

	V = 0
	policy = 0

	if args.algorithm == 'vi':
		V = valueIteration(numStates, numActions, transition, reward, gamma)
		Q = actionValueFunction(numStates, numActions, transition, reward, gamma, V)
		policy = np.argmax(Q, axis = 1)
		V = valueFunction(numStates, numActions, transition, reward, gamma, policy)

	elif args.algorithm == 'hpi':
		V, policy = howardPolicyIteration(numStates, numActions, transition, reward, gamma)

	elif args.algorithm == 'lp':
		V = linearProgrammingSolver(numStates, numActions, transition, reward, gamma)
		Q = actionValueFunction(numStates, numActions, transition, reward, gamma, V)
		policy = np.argmax(Q, axis = 1)
		V = valueFunction(numStates, numActions, transition, reward, gamma, policy)

	for i in range(numStates):
		print('{:.6f} {}'.format(V[i], policy[i]))
