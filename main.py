from board import Board
import minimax as Minimax
import TicTacToe_RL as RL
import random
import matplotlib.pyplot as plt


TRIALS = 5
ROUNDS = 10

q_table = RL.json_load(3)

for t in range(TRIALS):
  roundWins = []
  print("=" * 30 + "Trial " + str(t+1) + "=" * 30)
  
  for r in range(ROUNDS):
    who_first = random.random()

    # 1 = Minimax, 0 = RL
    if who_first > 0.5:
      board = Board(size=3)
    else:
      board = Board(size=3, new_turn=0)

    board.isWon()
    while True:
      if board.turn == 1:
        board = Minimax.minimax(board, 10)
      else:
        board = RL.choose_best_move(q_table, board, False)

      if board.won:
        if board.turn == 1:
          #print("RL won!")
          roundWins.append("R")
        else:
          #print("Minimax won!")
          roundWins.append("M")
        break
      elif board.full and board.won is False:
        #print("It's a tie!")
        roundWins.append("T")
        break
  
  print("Minimax Wins: " + str(roundWins.count("M")))
  print("RL Wins: " + str(roundWins.count("R")))
  print("Tied Games: " + str(roundWins.count("T")))
  print("=" * 67)
    
"""
from board import Board
import minimax as Minimax
import TicTacToe_RL as RL
import random



curr_board = Board(3, None)
q_table = RL.json_load(curr_board.size)

rand_turn = random.random()

while curr_board.won is False and not curr_board.full:
    if rand_turn > 0.5:
      print("Minimax Player: " + str(curr_board.turn) + "'s Move")
      curr_board = Minimax.minimax(curr_board, 10)
    else:
      print("RL Player: " + str(curr_board.turn) + "'s Move")
      curr_board = RL.choose_best_move(q_table, curr_board, False)
    print("Board")
    print(curr_board)
    if curr_board.won is True:
      if rand_turn > 0.5:
        print("Minimax won")
      else:
        print("RL won")
    elif not curr_board.full:
      if rand_turn > 0.5:
        print("RL Player: " + str(curr_board.turn) + "'s Move")
        curr_board = RL.choose_best_move(q_table, curr_board, False)
      else:
        print("Minimax Player: " + str(curr_board.turn) + "'s Move")
        curr_board = Minimax.minimax(curr_board, 10)
      print("Board")
      print(curr_board)
      if curr_board.won is True:
        if rand_turn > 0.5:
          print("RL won")
        else:
          print("Minimax won")
      elif curr_board.full:
        print("Its a tie!!!")
    else:
print("Its a tie!!")
"""