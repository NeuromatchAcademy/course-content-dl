class RandomPlayer():
  """
  Simulates Random Player
  """

  def __init__(self, game):
    self.game = game

  def play(self, board):
    """
    Simulates game play

    Args:
      board: np.ndarray
        Board of size n x n [6x6 in this case]

    Returns:
      a: int
        Randomly chosen move
    """

    # Compute the valid moves using getValidMoves()
    valids = self.game.getValidMoves(board, 1)

    # Compute the probability of each move being played (random player means this should
    # be uniform for valid moves, 0 for others)
    prob = valids/valids.sum()

    # Pick a random action based on the probabilities (hint: np.choice is useful)
    a = np.random.choice(self.game.getActionSize(), p=prob)

    return a
