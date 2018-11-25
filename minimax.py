from board import Board
import numpy as np
from random import shuffle

DEPTH=3

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



        