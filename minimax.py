from board import Board
import numpy as np
from random import shuffle

def minimax(s):
    def minimax_value(state, maximize):
        #This game is a win
        if state.won and maximize:
            return 1

        #This game is a lose
        elif state.won and not maximize:
            return -1

        #The game is a draw so return 0
        elif state.full and state.won is False:
            return 0

        #Generate successors
        successors = state.moves()

        #If MAX is to move
        if maximize:
            value = -9999
            for s in successors:
                value = max(value, minimax_value(s, False))
        else:
            value = 9999
            for s in successors:
                value = min(value, minimax_value(s, True))
        
        return value
    
    #For all successors of the current state, evaluate their minimax values
    successors = s.moves()
    scores = []
    for i in successors:
        scores.append(minimax_value(i, True))

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

while test.won is False:
    test = minimax(test)
    print("Player: " + str(test.turn) + "'s Move")
    print("Board")
    print(test)



        