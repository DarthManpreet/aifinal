from board import Board
import numpy as np
import random
np.set_printoptions(threshold=np.nan)

ETA = 1
GAMMA = 0.9

# Need a way to save qtable between runs. This would allow the AI
# to retain its knowledge and potentially play against human or
# other AI opponents. Possibly export to `.txt` file.
#
# May want to develop better reward scheme.
#
# Should play with ETA and GAMMA values
#
# Q-learning Neural Net if time permits. Would need a database of
# 4x4 games to learn from. Best if they were `expert` games.

# board_state = np.array([[0,0,0,0],
#                         [1,2,1,1],
#                         [1,2,1,1],
#                         [1,2,1,1]])

# board_init = np.array([[1,2],
#                        [2,2]])
#
# board0 = Board(2, None, 0)
# board1 = board0.moves()[0]
# board2 = board1.moves()[0]
# board3 = board1.moves()[1]
# board_restart = board0
#
# q_table = np.zeros((3**(board0.size * board0.size), 2*(board0.size * board0.size)))
# print q_table.shape

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
        action_idx = move_x + size * move_y
    else:
        action_idx = move_x + size * move_y + size*size
    # print state_idx, action_idx
    qtable_value = qtable[state_idx, action_idx]
    # print "current qtable value: ", qtable_value
    qtable_half = qtable.shape[1] / 2
    if next_turn == 0:
        next_reward = best_next_action(qtable[:, qtable_half:], next_state)
    else:
        next_reward = best_next_action(qtable[:, :qtable_half], next_state)
    qtable_update = qtable_value + ETA * (reward(next_state) + GAMMA * next_reward - qtable_value)
    qtable[state_idx, action_idx] = qtable_update
    return


def reward(state):
    temp_board = Board(len(state), state, None)
    if temp_board.won:
        # print temp_board
        # print "\n\n\nFound winning move\n\n\n! Reward: ", 10.0
        return 1.0
    else:
        return 0.0

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

# print board0
# print board1
# print board2
# print board0.state - board1.state
#
# print q_table
# update(q_table, board0.state, board1.state, board0.turn)
# print q_table
# update(q_table, board1.state, board2.state, board1.turn)
# print q_table
# update(q_table, board1.state, board3.state, board1.turn)
# print q_table
# update(q_table, board1.state, board2.state, board1.turn)
# print q_table
# update(q_table, board_restart.state, board1.state, board_restart.turn)
# print q_table

board = Board(3, None, 1)
q_table = np.zeros((3**(board.size * board.size), 2*(board.size * board.size)))
for i in range(100000):
    possible_moves = board.moves()
    random_move = random.randint(0, len(possible_moves)-1)
    next_board = possible_moves[random_move]
    update(q_table, board.state, next_board.state, board.turn)
    board = next_board
    if board.full or board.won:
        # print "\n...restarting...\n"
        board = Board(board.size, None, random.randint(0, 1))

print "Minimum", np.min(q_table[np.nonzero(q_table)])
