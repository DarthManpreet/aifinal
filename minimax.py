from board import Board
import numpy as np
from random import shuffle

def minimax(s):
    currentPlayer = s.turn

    def minimax_value(state):
        #This game is a win
        if state.won and state.turn == currentPlayer:
            return 1

        #This game is a lose
        elif state.won and state.turn != currentPlayer:
            return -1

        #The game is a draw so return 0
        elif state.full and state.won is False:
            return 0

        successors = state.moves()
        scores = []
        for s in successors:
            scores.append(minimax_value(s))
        
        if state.turn == currentPlayer:
            return max(scores)
        else:
            return min(scores)
    
    successors = s.moves()
    scores = []
    for i in successors:
        scores.append(minimax_value(i))

    print(scores)

    return successors[scores.index(max(scores))]
    

test = Board(size=3)
print("Player: " + str(test.turn) + "'s Move")
print("Board")
print(test)

while test.won is False:
    test = minimax(test)
    print("Player: " + str(test.turn) + "'s Move")
    print("Board")
    print(test)



        