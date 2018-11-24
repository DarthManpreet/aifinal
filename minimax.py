from board import Board
import numpy as np

def minimax_value(state):
    if state.won:
        return 9999
    elif state.move == 1:
        successors = state.moves()
        