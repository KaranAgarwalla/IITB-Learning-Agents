import numpy as np

class Model:

    def __init__(self, numRows, numCols, numMoves):
        self.numRows    = numRows
        self.numCols    = numCols
        self.numMoves   = numMoves
        self.numStates  = numRows * numCols
        self.T          = np.zeros((self.numStates, self.numMoves, self.numStates))
        self.R          = np.zeros((self.numStates, self.numMoves, self.numStates))
        self.total_T    = np.zeros((self.numStates, self.numMoves, self.numStates))
        self.total_R    = np.zeros((self.numStates, self.numMoves, self.numStates))
        self.total_V    = np.zeros((self.numStates, self.numMoves))
        self.visited    = []

    def updateModel(self, currState, a, nextState, r):

        s   = currState[0]*self.numCols + currState[1]
        ns  = nextState[0]*self.numCols + nextState[1]
        
        if self.total_V[s][a] == 0:
            self.visited.append((s, a))

        self.total_T[s][a][ns]  += 1
        self.total_R[s][a][ns]  += r
        self.total_V[s][a]      += 1
        self.T[s, a, :]         = self.total_T[s, a, :] / self.total_V[s][a]
        self.R[s, a, ns]        = self.total_R[s, a, ns] / self.total_T[s, a, ns]
    
    def obtainTransition(self):
        """
            Returns a randomly generated s, a, r, ns
        """

        index   = np.random.choice(len(self.visited))
        (s, a)  = self.visited[index]
        ns      = np.random.choice(self.numStates, p = self.T[s, a, :])
        r       = self.R[s, a, ns]
        return (int(s/self.numCols), s%self.numCols), a, (int(ns/self.numCols), ns%self.numCols), r