from board import Board
import numpy as np
from random import shuffle

DEPTH=20000

"""
def minimax(s, depth):
    def minimax_value(state, depth, maximize):
        #This game is a win
        #print("max value", maximize)
        if state.won and maximize and depth == 0:
            return 1

        #This game is a lose
        elif state.won and not maximize and depth == 0:
            return -1

        #The game is a draw so return 0
        elif state.full and state.won is False and depth == 0:
            return 0

        #print("received state", state)
        #Generate successors
        successors = state.moves()
        #print("Suucessors inside", successors)


        #If MAX is to move
        if maximize:
            value = -9999
            for s in successors:
                value = max(value, minimax_value(s, depth-1, False))
        else:
            value = 9999
            for s in successors:
                value = min(value, minimax_value(s, depth-1, True))
        #print("returning value", value)
        return value
    
    #For all successors of the current state, evaluate their minimax values
    successors = s.moves()
    #print("Suucessors", successors)
    scores = []
    for i in successors:
        #print("Now sending", i)
        scores.append(minimax_value(i, depth, True))

    #print("scores", scores)

    #Convert it into numpy array for tie breaking scenarios
    maxValue = max(scores)
    scores = np.array(scores)
    indices = np.argwhere(scores == maxValue).flatten()
    
    #Pick a random move if there's multiple moves with the same minimax value
    return successors[np.random.choice(indices)]
"""

def evaluate(s):
    score = 0
    opponent = int(not s.turn)

    for i in range(s.size):
        #Current row has no opponent pieces
        if np.count_nonzero(s.state[i, :] == opponent) == 0:
            #Check row for our markers
            if np.count_nonzero(s.state[i, :] == s.turn) == 1:
                score += 1
            elif np.count_nonzero(s.state[i, :] == s.turn) == 2:
                score += 50
            elif np.count_nonzero(s.state[i, :] == s.turn) == 3:
                score += 1000
    
        #Current row has none of my pieces
        if np.count_nonzero(s.state[i, :] == s.turn) == 0:
            #Check row for opponent markers
            if np.count_nonzero(s.state[i, :] == opponent) == 1:
                score -= 1
            elif np.count_nonzero(s.state[i, :] == opponent) == 2:
                score -= 50
            elif np.count_nonzero(s.state[i, :] == opponent) == 3:
                score -= 1000
        
        #Current column has no opponent pieces
        if np.count_nonzero(s.state[:, i] == opponent) == 0:
            #Check column for our markers
            if np.count_nonzero(s.state[:, i] == s.turn) == 1:
                score += 1
            elif np.count_nonzero(s.state[:, i] == s.turn) == 2:
                score += 50
            elif np.count_nonzero(s.state[:, i] == s.turn) == 3:
                score += 1000
        
        #Current column has none of my pieces
        if np.count_nonzero(s.state[:, i] == s.turn) == 0:
            #Check column for opponent markers
            if np.count_nonzero(s.state[:, i] == opponent) == 1:
                score -= 1
            elif np.count_nonzero(s.state[:, i] == opponent) == 2:
                score -= 50
            elif np.count_nonzero(s.state[:, i] == opponent) == 3:
                score -= 1000
        
    #Check diagonals for our markers
    if np.count_nonzero(np.diag(s.state) == opponent) == 0:
        if np.count_nonzero(np.diag(s.state) == s.turn) == 1:
            score += 1
        elif np.count_nonzero(np.diag(s.state) == s.turn) == 2:
            score += 50
        elif np.count_nonzero(np.diag(s.state) == s.turn) == 3:
            score += 1000

    #Check diagonals for opponent markers
    if np.count_nonzero(np.diag(s.state) == s.turn) == 0:
        if np.count_nonzero(np.diag(s.state) == opponent) == 1:
            score -= 1
        elif np.count_nonzero(np.diag(s.state) == opponent) == 2:
            score -= 50
        elif np.count_nonzero(np.diag(s.state) == opponent) == 3:
            score -= 1000
    
    #Check the other diagonal
    if np.count_nonzero(np.diag(np.fliplr(s.state)) == opponent) == 0:
        if np.count_nonzero(np.diag(np.fliplr(s.state)) == s.turn) == 1:
            score += 1
        elif np.count_nonzero(np.diag(np.fliplr(s.state)) == s.turn) == 2:
            score += 50
        elif np.count_nonzero(np.diag(np.fliplr(s.state)) == s.turn) == 3:
            score += 1000

    if np.count_nonzero(np.diag(np.fliplr(s.state)) == s.turn) == 0:
        if np.count_nonzero(np.diag(np.fliplr(s.state)) == opponent) == 1:
            score -= 1
        elif np.count_nonzero(np.diag(np.fliplr(s.state)) == opponent) == 2:
            score -= 50
        elif np.count_nonzero(np.diag(np.fliplr(s.state)) == opponent) == 3:
            score -= 1000

    return score

def minimax(s, depth):
    def minimax_value(state, depth, alpha, beta, maximize):
        if (depth == 0) or (state.won and maximize) or (state.won and not maximize) or (state.full and state.won is False):
            return evaluate(state)

        """
        if state.won and maximize:
            return 10000

        #This game is a lose
        elif state.won and not maximize:
            return -10000

        #The game is a draw so return 0
        elif state.full and state.won is False:
            return evaluate(state)
        """

        #Generate successors
        successors = state.moves()

        #If MAX is to move
        if maximize:
            value = -np.inf
            for s in successors:
                value = max(value, minimax_value(s, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = np.inf
            for s in successors:
                value = min(value, minimax_value(s, depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value
        #return value
    
    #For all successors of the current state, evaluate their minimax values
    successors = s.moves()
    scores = []
    for i in successors:
        scores.append(minimax_value(i, depth, -np.inf, np.Inf, True))

    print(scores)

    #Convert it into numpy array for tie breaking scenarios
    maxValue = max(scores)
    scores = np.array(scores)
    indices = np.argwhere(scores == maxValue).flatten()
    
    #Pick a random move if there's multiple moves with the same minimax value
    return successors[np.random.choice(indices)] 

#For testing purposes
test = Board(size=3)
print("Player: " + str(test.turn) + "'s Move")
print("Board")
print(test)

while test.won is False and not test.full:
    test = minimax(test, DEPTH)
    print("Player: " + str(test.turn) + "'s Move")
    print("Board")
    print(test)
