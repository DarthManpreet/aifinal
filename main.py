from board import Board
import minimax as Minimax
import TicTacToe_RL as RL
import random
import pandas as pd
import matplotlib.pyplot as plt

#Number of trials
TRIALS = 10

#Number of games player per trial
GAMES = 10

#Load Q-Table for RL
q_table = RL.json_load(3)

#Keep track of score
trialWins = {}
for t in range(TRIALS):
  roundWins = []
  print("=" * 30 + "Trial " + str(t+1) + "=" * 30)
  
  for r in range(GAMES):
    #50% that Minimax (playing as X) goes first
    #or RL (playing as O) goes first. 
    who_first = random.random()

    # 1 = Minimax, 0 = RL
    if who_first > 0.5:
      board = Board(size=3)
    else:
      board = Board(size=3, new_turn=0)

    #While the game is not over
    while True:
      #Minimax's move
      if board.turn == 1:
        board = Minimax.minimax(board, 10)
      else:
        #RL's move
        board = RL.choose_best_move(q_table, board, False)

      #Is the game over?
      board.isWon()
      if board.won:
        if board.turn == 1:
          roundWins.append("R")
        else:
          roundWins.append("M")
        break
      elif board.full and board.won is False:
        roundWins.append("T")
        break
  
  #Print summary for trial
  print("Minimax Wins: " + str(roundWins.count("M")))
  print("RL Wins: " + str(roundWins.count("R")))
  print("Tied Games: " + str(roundWins.count("T")))
  print("=" * 67)

  trialWins[t+1] = {"Minimax Wins": roundWins.count("M"), "RL Wins": roundWins.count("R"), "Tied": roundWins.count("T")}

#Calculate Average Games Won
m = 0
r = 0
t = 0

for index, rounds in trialWins.items():
  for key, value in rounds.items():
    if key == "Minimax Wins":
      m += value
    elif key == "RL Wins":
      r += value
    elif key == "Tied":
      t += value

print("=" * 30 + "Summary" + "=" * 30)
print("Average wins for Minimax: " + str(round(m / TRIALS, 2)))
print("Average wins for RL: " + str(round(r / TRIALS, 2)))
print("Average games tied: " + str(round(t/ TRIALS, 2)))
print("=" * 67)

#Generate bar graph of the score
pd.DataFrame(trialWins).T.plot(kind='bar')
plt.figure(1)
plt.title("Minimax vs Reinforcement Learning")
plt.legend()
plt.ylabel("Number of Games")
plt.yticks(range(0, GAMES + 1, 1))
plt.xticks(rotation=0)
plt.xlabel("Number of Trials")
plt.savefig("results.png")
plt.clf()