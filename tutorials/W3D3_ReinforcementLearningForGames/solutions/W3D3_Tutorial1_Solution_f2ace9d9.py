class PolicyBasedPlayer():
  def __init__(self, game, pnet, greedy=True):
    self.game = game
    self.pnet = pnet
    self.greedy = greedy

  def play(self, board):
    valids = self.game.getValidMoves(board, 1)
    action_probs = self.pnet.predict(board)
    vap = action_probs*valids  # masking invalid moves
    sum_vap = np.sum(vap)

    if sum_vap > 0:
      vap /= sum_vap  # renormalize
    else:
      # if all valid moves were masked we make all valid moves equally probable
      print("All valid moves were masked, doing a workaround.")
      vap = vap + valids
      vap /= np.sum(vap)

    if self.greedy:
      # greedy policy player
      a = np.where(vap == np.max(vap))[0][0]
    else:
      # sample-based policy player
      a = np.random.choice(self.game.getActionSize(), p=vap)

    return a


# playing games
num_games = 20
player1 = PolicyBasedPlayer(game, pnet, greedy=True).play
player2 = RandomPlayer(game).play
arena = Arena.Arena(player1, player2, game, display=OthelloGame.display)
## Uncomment below to test!
result = arena.playGames(num_games, verbose=False)
print(f"\n\n{result}")
win_rate_player1 = result[0] / num_games
print(f"\nWin rate for player1 over {num_games} games: {round(win_rate_player1*100, 1)}")