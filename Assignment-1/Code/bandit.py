from banditsim import BanditInstance
from utils import check_positive, restricted_float
from algorithms import EpsilonGreedy, UCB, KLUCB, ThompsonSampling, ThompsonSamplingWithHint
import numpy as np 
import argparse
import matplotlib.pyplot as plt

# instance from "../instances/i-1.txt"; "../instances/i-2.txt"; "../instances/i-3.txt",
# algorithm from epsilon-greedy with epsilon set to 0.02; ucb, kl-ucb, thompson-sampling,
# horizon from 100; 400; 1600; 6400; 25600; 102400, and random seed from 0; 1; ...; 49.

def history(means, algorithm, seed, horizon, epsilon=0.02):
	'''
		Returns regret history from T = 0 to T = horizon
	'''
	np.random.seed(seed)
	bandit 			= 	BanditInstance(means)
	optimal_arm 	= 	np.amax(means)
	reward_sum 		=	0
	regret 			= 	np.zeros(horizon+1)

	strategy		= 	0

	if algorithm   == 'epsilon-greedy':
		strategy	= EpsilonGreedy(means.shape[0], epsilon)

	elif algorithm == 'ucb':
		strategy	= UCB(means.shape[0])

	elif algorithm == 'kl-ucb':
		strategy	= KLUCB(means.shape[0])

	elif algorithm == 'thompson-sampling':
		strategy	= ThompsonSampling(means.shape[0])
	
	elif algorithm == 'thompson-sampling-with-hint':
		strategy	= ThompsonSamplingWithHint(means.shape[0], np.sort(means))

	for i in range(1, horizon+1):
		arm 		= 	strategy.getArm()
		reward 		= 	bandit.pull(arm)
		strategy.getReward(arm, reward)
		reward_sum += 	reward
		regret[i]	= 	i*optimal_arm - reward_sum

	return regret

def generate_T1():

	# T1
	T1 		   = open('outputDataT1.txt', 'w') 
	instances  = ["../Instances/I-1.txt", "../Instances/I-2.txt", "../Instances/I-3.txt"]
	algorithms = ['epsilon-greedy', 'ucb', 'kl-ucb', 'thompson-sampling']
	horizons   = [100, 400, 1600, 6400, 25600, 102400]
	seeds	   = np.arange(50)

	for index, instance in enumerate(instances):
		means = []
		with open(instance) as f:
			means = means + [line.rstrip('\n') for line in f]
		means = np.array(means, dtype=np.float64)

		plt.figure()
		for algorithm in algorithms:

			regret_sum = np.zeros((len(horizons),))
			for seed in seeds:
				cregret = history(means, algorithm, seed, horizons[-1])
				regret_sum += cregret[horizons]
				for horizon in horizons:
					print("{}, {}, {}, {}, {}, {}".format(instance, algorithm, seed, \
						0.02, horizon, cregret[horizon]), file = T1)

			average_regret = regret_sum/len(seeds)
			plt.plot(horizons, average_regret, label = f'{algorithm}')

		plt.legend()
		plt.xscale("log")
		plt.xlabel("Horizon")
		plt.ylabel("Regret")
		plt.title(f'T1-I-{index+1}')
		plt.savefig(f'T1-I-{index+1}.png')
		plt.close()

def generate_T2():

	# T2
	T2 		   = open('outputDataT2.txt', 'w') 
	instances  = ["../Instances/I-1.txt", "../Instances/I-2.txt", "../Instances/I-3.txt"]
	algorithms = ['thompson-sampling', 'thompson-sampling-with-hint']
	horizons   = [100, 400, 1600, 6400, 25600, 102400]
	seeds	   = np.arange(50)

	for index, instance in enumerate(instances):

		means = []
		with open(instance) as f:
			means = means + [line.rstrip('\n') for line in f]
		means = np.array(means, dtype=np.float64)

		plt.figure()
		for algorithm in algorithms:

			regret_sum = np.zeros((len(horizons),))
			for seed in seeds:
				cregret = history(means, algorithm, seed, horizons[-1])
				regret_sum += cregret[horizons]
				for horizon in horizons:
					print("{}, {}, {}, {}, {}, {}".format(instance, algorithm, seed, \
						0.02, horizon, cregret[horizon]), file = T2)

			average_regret = regret_sum/len(seeds)
			plt.plot(horizons, average_regret, label = f'{algorithm}')

		plt.legend()
		plt.xscale("log")
		plt.xlabel("Horizon")
		plt.ylabel("Regret")
		plt.title(f'T2-I-{index+1}')
		plt.savefig(f'T2-I-{index+1}.png')
		plt.close()

def generate_T3():

	# Generate Regret vs Epsilon for all three instances
	instances  = ["../Instances/I-1.txt", "../Instances/I-2.txt", "../Instances/I-3.txt"]
	algorithms = ['epsilon-greedy']
	horizons   = [102400]
	seeds	   = np.arange(50)
	epsilon    = [0.001, 0.01, 0.1]
	def average_regret(instance):
		means = []
		with open(instance) as f:
			means = means + [line.rstrip('\n') for line in f]
		means = np.array(means, dtype=np.float64)

		regret = np.zeros((3, ))
		for index in np.arange(3):
			regret_sum = 0
			for seed in seeds:
				regret_sum += history(means, algorithms[0], seed, horizons[0], epsilon[index])[-1]
			regret_sum /= len(seeds)
			regret[index] = regret_sum
		return regret

	for index, instance in enumerate(instances):
			print(average_regret(instance))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--instance", help="Path to the instance file")
	parser.add_argument("--algorithm", choices = ['epsilon-greedy', 'ucb', 'kl-ucb', \
		'thompson-sampling', 'thompson-sampling-with-hint'])
	parser.add_argument("--randomSeed", type = check_positive, help="seed")
	parser.add_argument("--epsilon", type = restricted_float, default = 0.02, help="epsilon for \
		epsilon-greedy algorithm")
	parser.add_argument("--horizon", type = check_positive, help = "Time Horizon for simulation")
	parser.add_argument("--generate_T1", action='store_true', help = "Generate Data for Task1")
	parser.add_argument("--generate_T2", action='store_true', help = "Generate Data for Task2")
	parser.add_argument("--epsilon_plots", action='store_true', help='Regret vs Epsilon')
	args = parser.parse_args()

	if args.generate_T1:
		generate_T1()

	elif args.generate_T2:
		generate_T2()

	elif args.epsilon_plots:
		generate_T3()
	else:
		means = []
		with open(args.instance) as f:
			means = means + [line.rstrip('\n') for line in f]
		means = np.array(means, dtype=np.float64)
		regret = history(means, args.algorithm, args.randomSeed, args.horizon, args.epsilon)

		print("{}, {}, {}, {}, {}, {}".format(args.instance, args.algorithm, args.randomSeed, \
			args.epsilon, args.horizon, regret[args.horizon]))

