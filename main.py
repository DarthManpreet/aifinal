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