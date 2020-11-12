
## Assignment 3: TD Control

#### To run all the tasks:
```
      ./run.sh
```

Below is description of usage of main.py. 
```
        python3 main.py --help
	    
	    usage: main.py [-h] [--numRows NUMROWS] [--numCols NUMCOLS]
               [--windStrength WINDSTRENGTH] [--Stochastic]
               [--startState STARTSTATE] [--endState ENDSTATE]
               [--numMoves {4,8,9}]
               [--algorithm {Sarsa,QLearning,ExpectedSarsa,all}]
               [--epsilon EPSILON] [--learning_rate LEARNING_RATE]
               [--episodes EPISODES] [--numSeeds NUMSEEDS] [--showPath]
               [--taskID TASKID] [--title TITLE]

		optional arguments:
		  -h, --help            show this help message and exit
		  --numRows NUMROWS     Number of rows in grid
		  --numCols NUMCOLS     Number of columns in grid
		  --windStrength WINDSTRENGTH
		                        Enter wind strength as string
		  --Stochastic          Stochastic Grid
		  --startState STARTSTATE
		                        Enter start state as string
		  --endState ENDSTATE   Enter end state as string
		  --numMoves {4,8,9}    Baseline, King's Moves, 9 Moves
		  --algorithm {Sarsa,QLearning,ExpectedSarsa,all}
		  --epsilon EPSILON     epsilon
		  --learning_rate LEARNING_RATE
		                        Learning Rate
		  --episodes EPISODES   Number of episodes
		  --numSeeds NUMSEEDS   Number of seeds to average across
		  --showPath            Show Greedy Path
		  --taskID TASKID       Task Number
		  --title TITLE         Title of the plot
  ```

As a reference look at **run.sh** script.
The default settings are as specified in the textbook Example 6.5.

> **Note:** You may encounter RuntimeError with message 'In A Cycle for Optimal Path' in some settings due to optimal policy being stuck in a cycle.
