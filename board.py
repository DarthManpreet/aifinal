#This file contains the implementation for the 4x4 tic-tac-toe board

import numpy as np

class Board:
    """
    This class defines the board for 4x4 tic-tac toe.
    """
    def __init__(self, new_state=None, new_turn=None):
        """
        This method initializes the class
        Parameters:
            1) new_state = Set the board to the new state provided
            2) new_turn = Whose turn is it?
        """
        #If there's no new state, start with an empty board
        if new_state is None:
            #0 represents O's, 1 represents X's, and 2 represents empty squares
            self.state = np.array([[2,2,2,2], [2,2,2,2], [2,2,2,2], [2,2,2,2]])
        else:
            self.state = new_state

        #Keeps track of which player's move it is (i.e. 1 = Player 1 and 2 = Player 2)
        if new_turn is None:
            self.turn = 1
        else:
            self.turn = new_turn

        #Keeps track of whether the game is over
        self.finished = False
    
    def moves(self):
        """
        This method generates the list of possible moves that the
        current player can make. This is done by determining the indices
        of all the empty squares (i.e. squares with number 2). 
        """
        #List to hold all the possible moves
        possible_moves = []
        
        #Determine the indices of all empty squares
        empty_squares = np.argwhere(self.state == 2)

        #Whose turn is it next?
        next_turn = None

        #Generate new board states given the current board state and who's move it is
        for row,column in empty_squares:
            #Deep copy of the current state
            move = np.copy(self.state)

            #If it's player's 1 move, then mark with a X and mark turn as player 2's
            if self.turn == 1:
                move[row][column] = 1
                next_turn = 2
            else:
                move[row][column] = 0 #Otherwise mark with a O and mark turn as player 1's
                next_turn = 1

            possible_moves.append(Board(new_state=move,new_turn=next_turn))
        return possible_moves
    

        