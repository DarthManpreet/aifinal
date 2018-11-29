# Algorithm which trains an AI to play Tic-Tac-Toe. The AI is
# trained using Reinforcement learning.

from board import Board
from copy import deepcopy
import numpy as np
import random
import json

#Learning Rate
ETA = 0.75

# Discount Factor
GAMMA = 0.9

# Size of Tic-Tac-Toe board
BOARDSIZE = 3

# Number of total moved played during training
ITERATIONS = 200000

# Probability out of 100 of a random move during training
EPSILON = 5

# Number of self-play games to play after training
NUMSELFPLAY = 10000

def json_save(table, size):
    """
    Saves qtable data in a local file, Local file name depends on size of the
    board.This allows the AI to retain the lessons it's learned across multiple
    sessions
    :param table: qtable which holds state/action values implemented as a
                  2d-numpy array
    :param size: size of board
    :return:
    """

    # json requires serializable data, which a numpy array isn't. Thus,
    # converts the qtable into a python list first, then saves it.
    try:
        if size == 2:
            with open('qtable2x2.txt', 'w') as outfile:
                json.dump(table.tolist(), outfile)
        elif size == 3:
            with open('qtable3x3.txt', 'w') as outfile:
                json.dump(table.tolist(), outfile)
        else:
            with open('qtable4x4.txt', 'w') as outfile:
                json.dump(table.tolist(), outfile)
    except:
        print("Save fail")


def json_load(size):
    """
    Loads qtable data from a local file. Local file choice depends on
    size of the board. This qtable stores previously learned lessons
    :param size: size of board
    :return: qtable implemented as a 2D numpy array.
    """
    try:
        if size == 2:
            with open('qtable2x2.txt') as json_file:
                table = json.load(json_file)
        elif size == 3:
            with open('qtable3x3.txt') as json_file:
                table = json.load(json_file)
        else:
            with open('qtable4x4.txt') as json_file:
                table = json.load(json_file)
        print("Found File")
    except:
        # If there's no previously stored qtable then create a new one. The number
        # of rows is how many possible board configurations there are, and the
        # number of columns is how many possible actions can be made in a given
        # state. The number of states is calculated as follows:
        # Each square on the board can either have an `x` an `o` or a `blank` in it.
        # That is, there are 3 options for each square. As there are `size * size`
        # number of squares on the board, this means there are 3 ** (size * size)
        # possible unique configurations.
        # Finally each player can play in any of the `size * size` number of squares
        # and thus there are 2 * (size * size) number of possible actions.
        table = np.zeros((3**(size * size), 2*(size * size)))
        print("File Not Found")

    # Converts the python list back into a numpy array.
    return np.array(table)


def update(qtable, state, next_state, next_turn):
    """
    Updates qtable entries using the temporal difference formula.
    This allows the AI to learn from its experiences
    :param qtable: Holds q-value state/action values. Represented
                   using a 2d-numpy array
    :param state: State of board
    :param next_state: Board after the current player makes their move
    :param next_turn: Player who will be playing on next_state
    :return: No value returned, qtable updated via side-effect.
    """

    # Calculate unique index for the given state
    state_idx = state_idx_calc(state)
    size = state.shape[0]

    # Given the two input states, calculates what action was taken
    # by taking the difference between the two and finding where
    # the value is non-zero.
    action = state - next_state
    move_location = np.nonzero(action)

    # Given what the action is, finds the x and y coordinates on the
    # board of where the action took place.
    move_x = move_location[0][0]
    move_y = move_location[1][0]

    # action[move_x, move_y] is where the action took place, and thus
    # if the value at this location is 2, then that means an `o` must
    # have been played. That's because the underlying board has
    # `blank` == 2
    # `x`     == 1
    # `o`     == 0
    #  Thus, if the difference is 2, then it must be that it went from
    #  a `blank` == 2 to an `o` == 0.
    #
    # Now, we know that, for a given player, the available actions they
    # can make is changing any blank square to a square that has their
    # symbol in it; namely, an `o` or an `x`. Thus, they have a maximum
    # of `size * size` actions they can make. This calculates a unique
    # index for each action. Specifically, if a 3x3 board is used then
    # it calculates a number from 0-8 for `o` moves, and number from
    # 9-17 for `x` moves. Thus, the first half of the columns of the
    # qtable has `o` moves, while the second half has `x` moves.
    if action[move_x, move_y] == 2:
        action_idx = size * move_x + move_y
    else:
        action_idx = size * move_x + move_y + size*size

    # retreive qtable value for taking this action in the given
    # board configuration state.
    qtable_value = qtable[state_idx, action_idx]

    # Player `x` actions are stored in the 2nd half of the columns
    # of the qtable. Thus, if it's player `x` turn next, then send
    # in only these columns of the qtable, and calculate the best
    # action that can be taken among these columns. If it's player
    # `o` turn next, then send in the first half.
    #
    # Double divide symbol used to force qtable_half to be integer type.
    qtable_half = qtable.shape[1] // 2
    if next_turn == 1:
        next_reward = best_next_action(qtable[:, qtable_half:], next_state)
    else:
        next_reward = best_next_action(qtable[:, :qtable_half], next_state)

    # Computes temporal difference formula.
    # We calculate reward of next_state and next_turn, as next_state is the
    # configuration of the board after having made the move we decided to make.
    qtable_update = qtable_value + ETA * (reward(next_state, next_turn) + GAMMA * next_reward - qtable_value)

    # Updates qtable with the temporal difference.
    qtable[state_idx, action_idx] = qtable_update
    return


def reward(state, turn):
    """
    Calculates the reward for a given action. Only two rewards are given,
    and with equal weight. A positive reward for playing a winning move,
    and a negative reward if, after playing your move, the opponent can
    play a winning move.
    :param state: Board configuration after the player's move has been made
    :param turn: Variable corresponding to whose turn it is. 1 means `x`'s
                 turn, `0` means `o` turn.
    :return: Reward for the given action performed.
    """
    # Put board state into a class to take advantage of certain
    # class function calls.
    temp_board = Board(len(state), state, turn)

    # If we made a winning move, receive a positive reward.
    if temp_board.won:
        return 10.0
    else:
        # Checks to see if the opponent can play a winning
        # move after we played our move.
        oppo_moves = temp_board.moves()
        oppo_wins = False
        for i in range(len(oppo_moves)):
            oppo_moves[i].isWon()
            if oppo_moves[i].won == True:
                oppo_wins = True
        # If the opponent could play a winning then we
        # receive a negative reward
        if oppo_wins:
            return -10.0
        # Otherwise, no reward is given.
        else:
            return 0.0


def state_idx_calc(state):
    """
    Calculates a unique index for each board configuration. Each square
    on the board can either have an `x` an `o` or a `blank` in it. That
    is, there are 3 options for each square. As there are `size * size`
    number of squares on the board, this means there are 3**(size * size)
    possible configurations. Thus, we calculate the unique index for each
    configuration by using unique powers of 3 for each square, and assign
    coefficients as follows:
    `blank` = 0
    `o`     = 1
    `x`     = 2
    Thus, the following configuration
    x o x
    - - x
    - - o
    would have the following unique index:
    (2 * 3^0) + (1 * 3^1) + (2 * 3^2) +
    (0 * 3^3) + (0 * 3^4) + (2 * 3^5) +
    (0 * 3^6) + (0 * 3^7) + (1 * 3^8)
    These unique indices are used to index into the qtable to both update
    the q-values, as well as retrieve the best action for a given state.
    :param state: Given board configuration
    :return: returns an index into the q_table
    """
    state_idx = 0
    size = state.shape[0]
    for i in range(size):
        for j in range(size):
            if state[i][j] == 0:
                state_idx += 1 * 3**(i + (size * j))
            elif state[i][j] == 1:
                state_idx += 2 * 3**(i + (size * j))
    return state_idx


def best_next_action(qtable, state):
    """
    Calculates action which gives the best reward in the given state.
    Used in `update` function to see what the best future reward will
    after an action is taken. Different from `choose_best_move` function
    below in that it doesn't actually change the board configuration or
    take any action, it just returns a value from the qtable.
    :param qtable: Holds q-values for state/action pairs. In this case
                   it only contains the half of qtable which corresponds
                   to which players turn it is. That is, the qtable holds
                   the lessons learned by both players. However, a given
                   board configuration looks good or bad depending on whose
                   turn it is. Thus, the qtable passed in only contains
                   actions which correspond to which players turn it is.
    :param state: given board configuration.
    :return: Index of the best action learned so far.
    """

    # Calculate unique index for the given state
    state_idx = state_idx_calc(state)

    # Find the maximum (aka best action) along the row of possible actions
    return np.max(qtable[state_idx])

def choose_best_move(qtable, board, output_decision_process):
    """
    Function used for the AI to play what it has learned as the best
    available move. Does this by creating a deepcopy of `board`, making
    the move on this copy, and then returning the copy from the function.
    :param qtable: Holds qvalues for the state/action pairs.
    :param board: Current board configuration.
    :param output_decision_process: Bool used to specify whether AI should print
                                    out its moves and "thought process" to the
                                    screen.
    :return: New board with the new action played on it.
    """
    # Calculate unique index for the given state
    state_idx = state_idx_calc(board.state)

    # Looks up the learned rewards in the qtable for the given
    # state/action paris.
    qtable_half = qtable.shape[1] // 2
    if board.turn == 1:
        learned_rewards = qtable[state_idx, qtable_half:]
    else:
        learned_rewards = qtable[state_idx, :qtable_half]

    # Finds indices of which squares are available to play in.
    available_squares = np.argwhere(board.state == 2)

    # converts these indices into the action_idx values that are used
    # to store the qvalues in the qtables. It works as follows:
    # np.argwhere returns a list of tuples, where each tuple [x,y]
    # is the x and y coordinates on the board which have available spots to
    # play in. We convert this list of tuples into a list of "action indices"
    # by mapping an index converter along the list. For example if:
    #
    # available_squares = [ [0,0], [1,2] ], then after the mapping it becomes
    #
    # available_squares = [ 0, 5 ].
    #
    # These are exactly the indices we decided each action
    # should be assigned to for each action the players could make.
    available_squares = np.array(list(map(lambda x: x[1] + board.size * x[0], available_squares)))

    # Calculates which actions maximize the learned rewards.
    max_squares = np.where(learned_rewards == np.max(learned_rewards))[0]

    # Then calculates the best actions to take among the available actions
    best_squares = np.intersect1d(available_squares, max_squares)

    # If the intersection is non-zero then that means there are good moves
    # to play. Choose a random move among the best moves available.
    if len(best_squares) != 0:
        best_move = random.choice(best_squares)

    # If there are no best moves, then play any available move. This can
    # happen because the qvalues in the qtable are all initialized to 0.
    # However, when the player plays a losing move it receives a negative
    # reward. Thus, if the player gets in a situation where it knows, no
    # matter where it moves it will lose, then this means every available
    # move has a negative score. Thus, the maximum move will be a square
    # score 0, which may not be an available move to play. Thus, just make
    # any available move instead.
    else:
        best_move = random.choice(available_squares)

    # Once the best move is decided among the qvalues, convert this
    # back into x,y coordinates on the board. We do this, deepcopy
    # input board, and then make the move on the new board.
    move_x, move_y = divmod(best_move, board.size)
    temp_board = deepcopy(board)
    temp_board.state[move_x][move_y] = temp_board.turn

    # update whose turn it is.
    if temp_board.turn == 0:
        temp_board.turn = 1
    else:
        temp_board.turn = 0

    # Print out decsion process
    if output_decision_process:
        print("move options", learned_rewards)
        print("available squares", available_squares)
        print("max squares", max_squares)
        print("best squares", best_squares)
        print("best move", best_move)
        print(temp_board)

    # return updated board
    return temp_board


###############################
# Training phase of algorithm #
###############################

# init board, with the size of the board being given as a hyper
# parameter.
# tttboard = Board(BOARDSIZE, None, 0)
#
# # load qtable
# q_table = json_load(tttboard.size)
#
# # Make ITERATIONS number of moves.
# for i in range(ITERATIONS):
#
#     # With EPSILON probability make a random move.
#     if random.randint(0, 100) < EPSILON:
#         possible_moves = tttboard.moves()
#         random_move = random.randint(0, len(possible_moves)-1)
#         next_board = possible_moves[random_move]
#
#     # Otherwise, make the best move available.
#     else:
#         next_board = choose_best_move(q_table, tttboard, False)
#
#     # Update qtable after having made the move.
#     update(q_table, tttboard.state, next_board.state, next_board.turn)
#
#     # Update board configuration to the board containing the new move.
#     tttboard = next_board
#
#     # Check if the board configuration is in a won state
#     tttboard.isWon()
#
#     # If the board is full or in a won state then start the game
#     # over again, with the player who goes first being chosen at random.
#     if np.count_nonzero(tttboard.state == 2) == 0 or tttboard.won:
#         tttboard = Board(tttboard.size, None, random.randint(0, 1))
#
# print("Finished Training")
#
# json_save(q_table, tttboard.size)
#
# print("Finished Saving\nBeginning Self Play")
#
# ###################################################
# # Demonstration of the AI playing against itself  #
# # ----------------------------------------------  #
# #                                                 #
# # Change 3rd parameter of choose_best_move from   #
# # False to True if you'd like to view the actions #
# # of the AI, and it's thought process.            #
# ###################################################
#
# num_games = 0
# num_first_player_wins = 0
# num_second_player_wins = 0
# num_ties = 0
# first_player = random.randint(0, 1)
# board_play = Board(BOARDSIZE, None, first_player)
#
# # Have the AI play itself NUMSELFPLAY number of games to
# # demonstrate what the AI has learned. Each move made
# # is the best known move available in the position.
# while num_games < NUMSELFPLAY:
#
#     # Make best move in given position
#     board_play = choose_best_move(q_table, board_play, False)
#
#     # Check whether the board is in a winning state
#     board_play.isWon()
#
#     # If it is then increment number of games player, who won,
#     # and restart game with a random first player.
#     if board_play.won:
#         num_games += 1
#         # After a move is played the board turn flips. Thus, if
#         # the current turn is back to the first player, and the
#         # board state is in a win, this means the second player
#         # won.
#         if first_player == board_play.turn:
#             num_second_player_wins += 1
#         # If not, then the first player must have won.
#         else:
#             num_first_player_wins += 1
#         # Choose a random first player and start game over.
#         first_player = random.randint(0, 1)
#         board_play = Board(BOARDSIZE, None, first_player)
#     # If the board is full then increment games played,
#     # number of ties, and start another game with a random
#     # first player.
#     if np.count_nonzero(board_play.state == 2) == 0:
#         num_games += 1
#         num_ties += 1
#         first_player = random.randint(0, 1)
#         board_play = Board(BOARDSIZE, None, first_player)
#
# # Print out results of self-play
# print("Ties: ", num_ties)
# print("First Player Wins: ", num_first_player_wins)
# print("Second Player Wins: ", num_second_player_wins)
