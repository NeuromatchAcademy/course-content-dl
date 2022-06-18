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
    valids = self.game.getValidMoves(board, 1)
    prob = valids/valids.sum()
    a = np.random.choice(self.game.getActionSize(), p=prob)

    return a


# Add event to airtable
atform.add_event('Coding Exercise 1.1: Implement a random player')