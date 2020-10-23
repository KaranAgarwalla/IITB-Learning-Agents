import numpy as np 
import argparse
import matplotlib.pyplot as plt
import pulp

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--grid", help="Path to the gridfile")
	parser.add_argument("--value_policy", help="Value and Policy file")
	args = parser.parse_args()

	maze = np.loadtxt(args.grid, dtype = np.int64)
	value_policy = np.loadtxt(args.value_policy)
	value, policy = value_policy[:, 0], value_policy[:, 1]
	policy = policy.astype('int64')

	n, m = maze.shape
	start_state = 0
	end_state = []
	idx_to_loc = {}
	idxs = np.zeros((n, m), dtype = np.int64)
	last_idx = 0

	for i in range(1, n-1):
		for j in range(1, m-1):
			if maze[i][j] != 1:
				idxs[i][j] = last_idx
				idx_to_loc[last_idx] = (i, j)
				last_idx += 1

			if maze[i][j] == 2:
				start_state = idxs[i][j]
			if maze[i][j] == 3:
				end_state.append(idxs[i][j])

	# if value[start_state] < 0:
	# 	print("Did Not Reach Terminal State")
	# 	print("Aborting")
	# 	exit(0)

	direction = {0:'N', 1:'S', 2:'W', 3:'E'}
	transition = {0:(-1, 0), 1:(1, 0), 2:(0, -1), 3:(0, 1)}

	curr_state = start_state
	while curr_state not in end_state:
		print(direction[policy[curr_state]], end = ' ')
		x, y = idx_to_loc[curr_state]
		if maze[x][y] == 1:
			print("Aborting")
			exit(0)
		x += transition[policy[curr_state]][0]
		y += transition[policy[curr_state]][1]
		curr_state = idxs[x][y]

