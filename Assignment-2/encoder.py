import numpy as np 
import argparse
import matplotlib.pyplot as plt
import pulp

def encode(maze):
	'''
		Encodes the maze into a MDP problem
	'''
	n, m = maze.shape

	# Indexes for non-walled states
	start_state = 0
	end_state = []
	idxs = np.zeros((n, m), dtype = np.int64)
	last_idx = 0
	
	for i in range(1, n-1):
		for j in range(1, m-1):
			if maze[i][j] != 1:
				idxs[i][j] = last_idx
				last_idx += 1

			if maze[i][j] == 2:
				start_state = idxs[i][j]
			if maze[i][j] == 3:
				end_state.append(idxs[i][j])

	# Start Printing
	print(f'numStates {last_idx}')
	print(f'numActions {4}')
	print(f'start {start_state}')
	print('end ', end='')
	print(*end_state)

	# Print Transitions
	transition = {0:(-1, 0), 1:(1, 0), 2:(0, -1), 3:(0, 1)}
	for i in range(1, n-1):
		for j in range(1, m-1):

			# No Outward Transitions
			if maze[i][j] == 3 or maze[i][j] == 1:
				continue
			
			for key, value in transition.items():
				x = i + value[0]
				y = j + value[1]
				r = 0
				if maze[x][y] == 1:
					r = -10*n*m
					x = i
					y = j
				elif maze[x][y] == 3:
					r = 10*n*m
				else:
					r = -2
				print(f'transition {idxs[i][j]} {key} {idxs[x][y]} {r} {1}')

	print('mdptype episodic')
	print(f'discount {1}')

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--grid", help="Path to the gridfile")
	args = parser.parse_args()

	maze = np.loadtxt(args.grid, dtype = np.int64)
	encode(maze)

