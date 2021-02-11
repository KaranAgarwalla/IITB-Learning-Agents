#!/bin/bash

python3 main.py --algorithm Sarsa --episodes 170 --numSeeds 50 --taskID 2 --numMoves 4 --showPath --title "Baseline Plot for Sarsa(0)"
python3 main.py --algorithm Sarsa --episodes 170 --numSeeds 50 --taskID 3 --numMoves 8 --showPath --title "Plot for King's Move"
python3 main.py --algorithm Sarsa --episodes 170 --numSeeds 50 --taskID 4 --numMoves 8 --Stochastic --showPath --title "Plot for King's Move"
python3 main.py --algorithm all   --episodes 170 --numSeeds 50 --taskID 5 --numMoves 4 --showPath --title "Baseline Plot for Comparison"

python3 main.py --algorithm DynaQ   --episodes 170 --numSeeds 50 --taskID 6 --numMoves 8 --Stochastic --planningSteps 0 --title "QLearning Plot"
python3 main.py --algorithm DynaQ   --episodes 170 --numSeeds 50 --taskID 7 --numMoves 8 --Stochastic --planningSteps 2 --title "Plot for N = 2"
python3 main.py --algorithm DynaQ   --episodes 170 --numSeeds 50 --taskID 8 --numMoves 8 --Stochastic --planningSteps 5 --title "Plot for N = 5"
python3 main.py --algorithm DynaQ   --episodes 170 --numSeeds 50 --taskID 9 --numMoves 8 --Stochastic --planningSteps 20 --title "Plot for N = 20"
