#!/bin/bash

python3 main.py --algorithm Sarsa --episodes 170 --numSeeds 50 --taskID 2 --numMoves 4 --showPath --title "Baseline Plot for Sarsa(0)"
python3 main.py --algorithm Sarsa --episodes 170 --numSeeds 50 --taskID 3 --numMoves 8 --showPath --title "Plot for King's Move"
python3 main.py --algorithm Sarsa --episodes 170 --numSeeds 50 --taskID 4 --numMoves 8 --Stochastic --showPath --title "Plot for King's Move"
python3 main.py --algorithm all   --episodes 170 --numSeeds 50 --taskID 5 --numMoves 4 --showPath --title "Baseline Plot for Comparison"