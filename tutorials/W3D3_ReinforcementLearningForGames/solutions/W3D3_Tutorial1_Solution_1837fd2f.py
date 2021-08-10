class ValueBasedPlayer():
  def __init__(self, game, vnet):
    self.game = game
    self.vnet = vnet

  def play(self, board):
    valids = self.game.getValidMoves(board, 1)
    candidates = []
    max_num_actions = 4
    va = np.where(valids)[0]
    va_list = va.tolist()
    random.shuffle(va_list)
    for a in va_list:
      # return next board state using getNextState() function
      nextBoard, _ = self.game.getNextState(board, 1, a)
      # predict the value of next state using value network
      value = self.vnet.predict(nextBoard)
      # add the value and the action as a tuple to the candidate lists, note that you might need to change the sign of the value based on the player
      candidates += [(-value, a)]

      if len(candidates) == max_num_actions:
        break

    candidates.sort()

    return candidates[0][1]


# playing games between a value-based player and a random player
set_seed(seed=SEED)
num_games = 20
player1 = ValueBasedPlayer(game, vnet).play
player2 = RandomPlayer(game).play
arena = Arena.Arena(player1, player2, game, display=OthelloGame.display)
## Uncomment the code below to check your code!
result = arena.playGames(num_games, verbose=False)
print(f"\n\n{result}")