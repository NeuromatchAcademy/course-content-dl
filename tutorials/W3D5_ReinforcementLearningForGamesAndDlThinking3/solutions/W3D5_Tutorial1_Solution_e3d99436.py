class PolicyBasedPlayer():

  def __init__(self, game, pnet, greedy=True):
    """
    Args:
      game: OthelloGame instance
        Instance of the OthelloGame class above;
      pnet: Policy Network instance
        Instance of the Policy Network class above
      greedy: Boolean
        If true, implement greedy approach
        Else, implement random sample policy based player
    """
    self.game = game
    self.pnet = pnet
    self.greedy = greedy

  def play(self, board):
    """
    Args:
      board: np.ndarray
        Board of size n x n [6x6 in this case]

    Returns:
      a: np.ndarray
        If greedy, implement greedy policy player
        Else, implement random sample policy based player
    """
    valids = self.game.getValidMoves(board, 1)
    action_probs = self.pnet.predict(board)
    vap = action_probs*valids  # Masking invalid moves
    sum_vap = np.sum(vap)

    if sum_vap > 0:
      vap /= sum_vap  # Renormalize
    else:
      # If all valid moves were masked we make all valid moves equally probable
      print("All valid moves were masked, doing a workaround.")
      vap = vap + valids
      vap /= np.sum(vap)

    if self.greedy:
      # Greedy policy player
      a = np.where(vap == np.max(vap))[0][0]
    else:
      # Sample-based policy player
      a = np.random.choice(self.game.getActionSize(), p=vap)

    return a


# Playing games
set_seed(seed=SEED)
num_games = 20
player1 = PolicyBasedPlayer(game, pnet, greedy=True).play
player2 = RandomPlayer(game).play
arena = Arena.Arena(player1, player2, game, display=OthelloGame.display)
## Uncomment below to test!
result = arena.playGames(num_games, verbose=False)

print(f"\nNumber of games won by player1 = {result[0]}, "
      f"Number of games won by player2 = {result[1]} out of {num_games} games")

win_rate_player1 = result[0] / num_games
print(f"\nWin rate for greedy policy player 1 (vs random player 2) over {num_games} games: {round(win_rate_player1*100, 1)}%")