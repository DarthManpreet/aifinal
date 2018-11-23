import numpy as np

class TTT:
    """A square Tic-Tac-Toe board, which supports playing moves and checking if
    the board is in a won state. The board is represented internally
    as a 2D numpy array of `1`s and `0`s for `x` and `o`, and `-1` for
    unplayed spots. Also allows for a specified length parameter which
    controls the size of the board."""
    def __init__(self, length):
        self.board = np.full((length, length), -1)
        self.length = length

    def __repr__(self):
        """Pretty prints board. Internally, the board uses `1` for `x`
        and `0` for `o`. However, the board is printed using the expected
        `x` and `o` characters."""
        boardstr = ""
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if self.board[i][j] == 0:
                    boardstr += "o "
                elif self.board[i][j] == 1:
                    boardstr += "x "
                else:
                    boardstr += "- "
            boardstr += "\n"
        return boardstr

    def isWon(self):
        """Checks if current board state is won by a player or not.
        First checks all rows and columns, then checks diagonal, and
        finally checks reverse diagonal. Does this by counting whether
        the total number of `x` or `o` is equal to 4 or not. If it is,
        return `True` as well as the player who won."""
        for i in range(self.board.shape[0]):
            if np.count_nonzero(self.board[i, :] == 0) == self.length or \
               np.count_nonzero(self.board[i, :] == 1) == self.length or \
               np.count_nonzero(self.board[:, i] == 0) == self.length or \
               np.count_nonzero(self.board[:, i] == 1) == self.length:
                return True, self.board[i][i]
        if np.count_nonzero(np.diag(self.board) == 0) == self.length or \
           np.count_nonzero(np.diag(self.board) == 1) == self.length:
            return True, self.board[0][0]
        if np.count_nonzero(np.diag(np.fliplr(self.board)) == 1) == self.length or \
           np.count_nonzero(np.diag(np.fliplr(self.board)) == 1) == self.length:
            return True, self.board[0][self.length -1]
        return False, -1

    def insert(self, move, x, y):
        """Inserts specified move, so long as the coordinates lay within the board,
        the space is available to be played in, and the specified move is an `x`
        or `o`."""
        if -1 < x < self.length and -1 < y < self.length:
            if self.board[x][y] == -1:
                if move == "o" or move == "0":
                    self.board[x][y] = "0"
                elif move == "x" or move == "1":
                    self.board[x][y] = "1"
                else:
                    print "Error, invalid move. Use `x` or `o` not " + str(move)
            else:
                print "Error, invalid move. Player already used this board location"
        else:
            print "Error, invalid coordinates for move " + str(x) + " " + str(y)
        return

test = TTT(4)
test.insert("o", 0, 0)
test.insert("o", 1, 1)
test.insert("o", 2, 2)
test.insert("x", 0, 3)
test.insert("x", 1, 2)
test.insert("x", 0, 1)
test.insert("x", 2, 3)
test.insert("o", 3, 2)
test.insert("o", 3, 3)
test.insert("o", 4, 4)
print test
print test.isWon()
