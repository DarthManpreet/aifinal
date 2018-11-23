#This file contains the implementation for the 4x4 tic-tac-toe board

import numpy as np

class Board:
    """
    This class defines the board for 4x4 tic-tac toe.
    """
    def __init__(self, size=None, new_state=None, new_turn=None):
        """
        This method initializes the class
        Parameters:
            1) new_state = Set the board to the new state provided
            2) new_turn = Whose turn is it?
        """
        #If there's no specified board size, start with a 4x4 board
        if size is None:
            self.size = 4
        else:
            self.size = size

        #If there's no new state, start with an empty board
        if new_state is None:
            #0 represents O's, 1 represents X's, and 2 represents empty squares
            self.state = np.full((self.size, self.size), 2)
        else:
            self.state = new_state

        #Keeps track of which player's move it is (i.e. 1 = Player 1 and 2 = Player 2)
        if new_turn is None:
            self.turn = 1
        else:
            self.turn = new_turn

        #Keeps track of whether the game is over
        self.finished = False

    def __repr__(self):
        """
        Pretty prints board. Internally, the board uses `1` for `x`
        and `0` for `o`. However, the board is printed using the expected
        `x` and `o` characters.
        """
        boardstr = ""
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                if self.state[i][j] == 0:
                    boardstr += "o "
                elif self.state[i][j] == 1:
                    boardstr += "x "
                else:
                    boardstr += "- "
            boardstr += "\n"
        return boardstr

    def isWon(self):
        """
        Checks if current board state is won by a player or not.
        First checks all rows and columns, then checks diagonal, and
        finally checks reverse diagonal. Does this by counting whether
        the total number of `x` or `o` is equal to `self.size` or not.
        If it is, update `self.finished` to `True` and return bool.
        """

        # Checks all rows and columns to see if the number of `0`'s and
        # `1`'s is equal to the size of the board. If it is, that means
        # the board is finished.
        for i in range(self.size):
            if np.count_nonzero(self.state[i, :] == 0) == self.size or \
               np.count_nonzero(self.state[i, :] == 1) == self.size or \
               np.count_nonzero(self.state[:, i] == 0) == self.size or \
               np.count_nonzero(self.state[:, i] == 1) == self.size:
                self.finished = True

        # Does the same check as above, but this time for the two
        # diagonals.
        if np.count_nonzero(np.diag(self.state) == 0) == self.size or \
           np.count_nonzero(np.diag(self.state) == 1) == self.size or \
           np.count_nonzero(np.diag(np.fliplr(self.state)) == 0) == self.size or \
           np.count_nonzero(np.diag(np.fliplr(self.state)) == 1) == self.size:
            self.finished = True

        return self.finished

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

            possible_moves.append(Board(size=self.size, new_state=move, new_turn=next_turn))
        return possible_moves
