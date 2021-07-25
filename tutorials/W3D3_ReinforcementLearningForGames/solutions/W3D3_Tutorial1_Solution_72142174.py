class RandomPlayer():
  def __init__(self, game):
    self.game = game

  def play(self, board):

    valids = self.game.getValidMoves(board, 1)
    prob = valids/valids.sum()
    a = np.random.choice(self.game.getActionSize(), p=prob)

    return a