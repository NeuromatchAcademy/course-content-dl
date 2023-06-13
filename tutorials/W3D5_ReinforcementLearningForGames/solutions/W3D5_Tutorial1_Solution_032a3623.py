class ValueBasedPlayer():
  """
  Simulate Value Based Player
  """

  def __init__(self, game, vnet):
    """
    Initialise value based player parameters

    Args:
      game: OthelloGame instance
        Instance of the OthelloGame class above;
      vnet: Value Network instance
        Instance of the Value Network class above;

    Returns:
      Nothing
    """
    self.game = game
    self.vnet = vnet

  def play(self, board):
    """
    Simulate game play

    Args:
      board: np.ndarray
        Board of size n x n [6x6 in this case]

    Returns:
      candidates: List
        Collection of tuples describing action and values of future predicted states
    """
    valids = self.game.getValidMoves(board, 1)
    candidates = []
    max_num_actions = 4
    va = np.where(valids)[0]
    va_list = va.tolist()
    random.shuffle(va_list)
    for a in va_list:
      # Return next board state using getNextState() function
      nextBoard, _ = self.game.getNextState(board, 1, a)
      # Predict the value of next state using value network
      value = self.vnet.predict(nextBoard)
      # Add the value and the action as a tuple to the candidate lists, note that you might need to change the sign of the value based on the player
      candidates += [(-value, a)]

      if len(candidates) == max_num_actions:
        break

    # Sort by the values
    candidates.sort()

    # Return action associated with highest value
    return candidates[0][1]


# Playing games between a value-based player and a random player
set_seed(seed=SEED)
num_games = 20
player1 = ValueBasedPlayer(game, vnet).play
player2 = RandomPlayer(game).play
arena = Arena.Arena(player1, player2, game, display=OthelloGame.display)

## Uncomment the code below to check your code!
## Compute win rate for the value-based player (player 1)
result = arena.playGames(num_games, verbose=False)
print(f"\nNumber of games won by player1 = {result[0]}, "
      f"Number of games won by player2 = {result[1]} out of {num_games} games")
win_rate_player1 = result[0]/num_games
print(f"\nWin rate for player1 over 20 games: {round(win_rate_player1*100, 1)}%")