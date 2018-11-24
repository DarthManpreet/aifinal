from board import Board
from copy import deepcopy
import numpy as np
import random
import json
np.set_printoptions(threshold=np.nan)

ETA = 1
GAMMA = 0.9
BOARDSIZE = 3
ITERATIONS = 50000
EPSILON = 10


def json_save(table, size):
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
        print "Save fail"


def json_load(size):
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
        print "found file"
    except:
        table = np.zeros((3**(size * size), 2*(size * size)))
        print "File not found"
    return table


def update(qtable, state, next_state, next_turn):
    state_idx = 0
    size = state.shape[0]
    for i in range(size):
        for j in range(size):
            if state[i][j] == 0:
                state_idx += 1 * 3**(i + (size * j))
            elif state[i][j] == 1:
                state_idx += 2 * 3**(i + (size * j))

    action = state - next_state
    move_location = np.nonzero(action)
    move_x = move_location[0][0]
    move_y = move_location[1][0]
    if action[move_x, move_y] == 2:
        action_idx = size * move_x + move_y
    else:
        action_idx = size * move_x + move_y + size*size
    # print state_idx, action_idx
    qtable_value = qtable[state_idx, action_idx]
    # print "current qtable value: ", qtable_value
    qtable_half = qtable.shape[1] / 2
    if next_turn == 1:
        next_reward = best_next_action(qtable[:, qtable_half:], next_state)
    else:
        next_reward = best_next_action(qtable[:, :qtable_half], next_state)
    qtable_update = qtable_value + ETA * (reward(next_state, next_turn) + GAMMA * next_reward - qtable_value)
    qtable[state_idx, action_idx] = qtable_update
    return


def reward(state, turn):
    temp_board = Board(len(state), state, turn)
    # old_temp_board = Board(len(old_state), old_state, turn)
    # print turn
    if temp_board.won:
        # print "I won!", temp_board
        return 10.0
    else:
        oppo_moves = temp_board.moves()
        oppo_wins = False
        for i in range(len(oppo_moves)):
            oppo_moves[i].isWon()
            if oppo_moves[i].won == True:
                # print "Originally\n", old_temp_board, "I moved\n", temp_board, "but he won :(\n", oppo_moves[i]
                oppo_wins = True
        if oppo_wins:
            return -10.0
        else:
            return 0.0

# def two_unblocked():
#     return


def best_next_action(qtable, state):
    state_idx = 0
    size = state.shape[0]
    for i in range(size):
        for j in range(size):
            if state[i][j] == 0:
                state_idx += 1 * 3**(i + (size * j))
            elif state[i][j] == 1:
                state_idx += 2 * 3**(i + (size * j))

    # print "Max reward for next state: ", np.max(qtable[state_idx])
    return np.max(qtable[state_idx])

def choose_best_move(qtable, board, Playing):
    state_idx = 0
    size = board.state.shape[0]
    for i in range(size):
        for j in range(size):
            if board.state[i][j] == 0:
                state_idx += 1 * 3**(i + (size * j))
            elif board.state[i][j] == 1:
                state_idx += 2 * 3**(i + (size * j))

    qtable_half = qtable.shape[1] / 2
    if board.turn == 1:
        move_options = qtable[state_idx, qtable_half:]
    else:
        move_options = qtable[state_idx, :qtable_half]

    def move_index(x):
        return x[1] + board.size * x[0]

    available_squares = np.argwhere(board.state == 2)
    available_squares = np.array(map(move_index, available_squares))
    max_squares = np.where(move_options == np.max(move_options))[0]
    best_squares = np.intersect1d(available_squares, max_squares)
    if len(best_squares) != 0:
        best_move = random.choice(best_squares)
    else:
        best_move = random.choice(available_squares)


    move_x, move_y = divmod(best_move, board.size)
    temp_board = deepcopy(board)
    temp_board.state[move_x][move_y] = temp_board.turn

    if temp_board.turn == 0:
        temp_board.turn = 1
    else:
        temp_board.turn = 0


    if Playing:
        print "move options", move_options
        print "available squares", available_squares
        print "max squares", max_squares
        print "best squares", best_squares
        print "best move", best_move
        print temp_board

    return temp_board


tttboard = Board(BOARDSIZE, None, 0)
# q_table = load_q_table(board.size)
q_table = np.array(json_load(tttboard.size))
# print q_table
for i in range(ITERATIONS):
    if random.randint(0, 100) < EPSILON:
    # if tttboard.turn == 0:
        possible_moves = tttboard.moves()
        random_move = random.randint(0, len(possible_moves)-1)
        next_board = possible_moves[random_move]
    else:
        next_board = choose_best_move(q_table, tttboard, False)
    update(q_table, tttboard.state, next_board.state, next_board.turn)
    tttboard = next_board
    tttboard.isWon()
    if np.count_nonzero(tttboard.state == 2) == 0 or tttboard.won:
        # print "\n...restarting...\n"
        tttboard = Board(tttboard.size, None, random.randint(0, 1))

print "Done Training"

json_save(q_table, tttboard.size)

print "Done saving"

board_play = Board(tttboard.size, None, 1)
num_games = 0
while num_games < 10:
    board_play = choose_best_move(q_table, board_play, True)
    board_play.isWon()
    if board_play.won:
        print "won!"
        num_games += 1
        board_play = Board(BOARDSIZE, None, random.randint(0, 1))
    if np.count_nonzero(board_play.state == 2) == 0:
        print "board full"
        num_games += 1
        board_play = Board(BOARDSIZE, None, random.randint(0, 1))


