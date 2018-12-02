from board import Board
import numpy as np
from random import shuffle

#The depth limit for minimax search
DEPTH=10

def evaluate(s, player):
    """
    The heurisitics evaluation for the current state, given the current player.
    The heurisitic adds 1 point if the current player has one marker in row, column,
    or diagonal and the other 2 squares are empty. It gives it 50 points if there's 2
    and infinity points if the game is a win. However, points are subtracted if the
    opponent has markers with empty squares with the points deducted based on the same
    scenarios as described for points given.
    
    Parameters:
        1) s = the state of the board
        2) player = the current player
    
    Returns:
        1) score = the score for that given state
    """
    score = 0
    opponent = int(not player)

    for i in range(s.size):
        #Current row has no opponent pieces
        if np.count_nonzero(s.state[i, :] == opponent) == 0:
            #Check row for our markers
            if np.count_nonzero(s.state[i, :] == player) == 1:
                score += 1
            elif np.count_nonzero(s.state[i, :] == player) == 2:
                score += 50
            elif np.count_nonzero(s.state[i, :] == player) == 3:
                score = np.Inf
    
        #Current row has none of my pieces
        if np.count_nonzero(s.state[i, :] == player) == 0:
            #Check row for opponent markers
            if np.count_nonzero(s.state[i, :] == opponent) == 1:
                score -= 1
            elif np.count_nonzero(s.state[i, :] == opponent) == 2:
                score -= 50
            elif np.count_nonzero(s.state[i, :] == opponent) == 3:
                score = -np.inf
        
        #Current column has no opponent pieces
        if np.count_nonzero(s.state[:, i] == opponent) == 0:
            #Check column for our markers
            if np.count_nonzero(s.state[:, i] == player) == 1:
                score += 1
            elif np.count_nonzero(s.state[:, i] == player) == 2:
                score += 50
            elif np.count_nonzero(s.state[:, i] == player) == 3:
                score = np.Inf
        
        #Current column has none of my pieces
        if np.count_nonzero(s.state[:, i] == player) == 0:
            #Check column for opponent markers
            if np.count_nonzero(s.state[:, i] == opponent) == 1:
                score -= 1
            elif np.count_nonzero(s.state[:, i] == opponent) == 2:
                score -= 50
            elif np.count_nonzero(s.state[:, i] == opponent) == 3:
                score = -np.inf
        
    #Check diagonals for our markers
    if np.count_nonzero(np.diag(s.state) == opponent) == 0:
        if np.count_nonzero(np.diag(s.state) == player) == 1:
            score += 1
        elif np.count_nonzero(np.diag(s.state) == player) == 2:
            score += 50
        elif np.count_nonzero(np.diag(s.state) == player) == 3:
            score = np.Inf

    #Check diagonals for opponent markers
    if np.count_nonzero(np.diag(s.state) == player) == 0:
        if np.count_nonzero(np.diag(s.state) == opponent) == 1:
            score -= 1
        elif np.count_nonzero(np.diag(s.state) == opponent) == 2:
            score -= 50
        elif np.count_nonzero(np.diag(s.state) == opponent) == 3:
            score = -np.inf
    
    #Check the other diagonal
    if np.count_nonzero(np.diag(np.fliplr(s.state)) == opponent) == 0:
        if np.count_nonzero(np.diag(np.fliplr(s.state)) == player) == 1:
            score += 1
        elif np.count_nonzero(np.diag(np.fliplr(s.state)) == player) == 2:
            score += 50
        elif np.count_nonzero(np.diag(np.fliplr(s.state)) == player) == 3:
            score = np.Inf

    if np.count_nonzero(np.diag(np.fliplr(s.state)) == player) == 0:
        if np.count_nonzero(np.diag(np.fliplr(s.state)) == opponent) == 1:
            score -= 1
        elif np.count_nonzero(np.diag(np.fliplr(s.state)) == opponent) == 2:
            score -= 50
        elif np.count_nonzero(np.diag(np.fliplr(s.state)) == opponent) == 3:
            score = -np.inf

    return score

def minimax(s, depth):
    """
    This method performs the minimax search given an initial state
    and the depth.
    
    Parameters:
        1) s = the initial state
        2) depth = the depth limit
    
    Returns:
        1) Best move to make given the initial state
    """
    player = s.turn

    def minimax_value(state, depth, alpha, beta, maximize):
        """
        Determines the value of the given state in order to decide
        which move the player should make next. This also implements
        alpha-beta pruning with a depth-limited search.

        Parameters:
            1) state = the state to examine
            2) depth = the current depth of the search
            3) alpha = the alpha value for alpha-beta pruning
            4) beta = the beta value for alpha-beta pruning
            5) maximize = keep track whether it's MAX's turn or MIN's turn
        
        Returns:
            1) The best score for the initial state
        """

        #Terminal test checks
        if (depth == 0) or (state.won) or (state.full):
            return evaluate(state, player)
            
        #Generate successors
        successors = state.moves()

        #If MAX is to move
        if maximize:
            value = evaluate(state, player)
            for s in successors:
                value = max(value, minimax_value(s, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return alpha
        else:
            #MIN's move
            value = evaluate(state, player)
            for s in successors:
                value = min(value, minimax_value(s, depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return beta
    
    #For all successors of the current state, evaluate their minimax values
    successors = s.moves()
    scores = []
    for i in successors:
        scores.append(minimax_value(i, depth, -np.inf, np.Inf, True))

    #Convert it into numpy array for tie breaking scenarios
    maxValue = max(scores)
    scores = np.array(scores)
    indices = np.argwhere(scores == maxValue).flatten()
    
    #Pick a random move if there's multiple moves with the same minimax value
    return successors[np.random.choice(indices)]


"""
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
"""