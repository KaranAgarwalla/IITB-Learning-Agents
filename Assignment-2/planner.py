import numpy as np 
import argparse
import matplotlib.pyplot as plt
import pulp

def read(file):
	numStates, numActions = 0, 0
	startState, endStates = 0, 0
	T, R = 0, 0
	mdptype, gamma = 0, 0

	with open(file) as f:
			for l in f:
				cl = l.strip('\n').split()

				if cl[0] == 'numStates':
					numStates = int(cl[1])
				
				elif cl[0] == 'numActions':
					numActions = int(cl[1])
					T = np.zeros((numStates, numActions, numStates), dtype=np.float64)
					R = np.zeros((numStates, numActions, numStates), dtype=np.float64)
				
				elif cl[0] == 'start':
					startState = int(cl[1])
				
				elif cl[0] == 'end':
					endStates = []
					for i in range(1, len(cl)):
						endStates.append(int(cl[i]))
					endStates = np.array(endStates)
				
				elif cl[0] == 'transition':
					s1, ac, s2, r, p = int(cl[1]), int(cl[2]), int(cl[3]), float(cl[4]), float(cl[5])
					T[s1][ac][s2] = p
					R[s1][ac][s2] = r
				
				elif cl[0] == 'mdptype':
					mdptype = cl[1]
				
				elif cl[0] == 'discount':
					gamma = float(cl[1])
				
				else:
					print("Aborting: Incorrect Format")
					exit(0)

	return T, R, gamma

def actionValueFunction(T, R, gamma, V):
	'''
		returns Q value function for given policy and valueFunction
	'''
	Q = np.sum(T*R, axis = 2) + gamma*T@V
	return Q

def valueFunction(T, R, gamma, policy):
	'''
		Returns the value function given the policy
	'''
	numStates = T.shape[0]
	T_policy = np.array([T[i, policy[i], :] for i in range(numStates)])
	R_policy = np.array([R[i, policy[i], :] for i in range(numStates)])
	A = np.identity(numStates) - gamma*T_policy
	B = np.sum(T_policy*R_policy, axis = 1)
	return np.linalg.pinv(A)@B

def valueIteration(T, R, gamma):
	'''
		Returns optimal policy using Value Iteration
	'''
	V = np.zeros(T.shape[0])
	tolerance = 1e-12
	diff = 1
	while diff>tolerance:
		nV = np.amax(actionValueFunction(T, R, gamma, V), axis = 1)
		diff = np.linalg.norm(nV-V)
		V = nV
	return V

def howardPolicyIteration(T, R, gamma):
	'''
		Implements Howard Policy Iteration to find optimal value function V
	'''
	policy = np.random.choice(T.shape[1], T.shape[0])
	V = valueFunction(T, R, gamma, policy)
	tolerance = 1e-12
	while True:
		Q = actionValueFunction(T, R, gamma, V)
		deltaV = np.amax(Q, axis = 1) - V
		if np.amax(deltaV) < tolerance:
			break
		nPolicy = np.argmax(Q, axis = 1)
		policy[deltaV >= tolerance] = nPolicy[deltaV >= tolerance]
		V = valueFunction(T, R, gamma, policy)

	return V, policy

def linearProgrammingSolver(T, R, gamma):
	'''
		Implements linear programming solution to find optimal value function V
	'''
	numStates, numActions = T.shape[0], T.shape[1]
	prob = pulp.LpProblem("valueFunction", pulp.LpMinimize)
	V = pulp.LpVariable.dicts('valueFunction', range(numStates))

	#Formulate Objective Function
	prob += pulp.lpSum([V[s] for s in range(numStates)])

	#Formulate constraints
	for s in range(numStates):
		for a in range(numActions):
			prob += pulp.lpSum([T[s, a, nS]*(R[s, a, nS] + gamma*V[nS]) \
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

	T, R, gamma = read(args.mdp)

	np.random.seed(0)
	V = 0
	policy = 0

	if args.algorithm == 'vi':
		V = valueIteration(T, R, gamma)
		Q = actionValueFunction(T, R, gamma, V)
		policy = np.argmax(Q, axis = 1)

	elif args.algorithm == 'hpi':
		V, policy = howardPolicyIteration(T, R, gamma)

	elif args.algorithm == 'lp':
		V = linearProgrammingSolver(T, R, gamma)
		Q = actionValueFunction(T, R, gamma, V)
		policy = np.argmax(Q, axis = 1)
		V = valueFunction(T, R, gamma, policy)

	for i in range(T.shape[0]):
		print('{:.6f} {}'.format(V[i], policy[i]))
