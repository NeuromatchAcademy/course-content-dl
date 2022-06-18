class MonteCarloTreeSearchBasedPlayer():
  """
  Simulate Player based on MCTS
  """

  def __init__(self, game, nnet, args):
    """
    Initialize parameters of MCTS

    Args:
      game: OthelloGame instance
        Instance of the OthelloGame class above;
      nnet: OthelloNet instance
        Instance of the OthelloNNet class above;
      args: dictionary
        Instantiates number of iterations and episodes, controls temperature threshold, queue length,
        arena, checkpointing, and neural network parameters:
        learning-rate: 0.001, dropout: 0.3, epochs: 10, batch_size: 64,
        num_channels: 512

    Returns:
      Nothing
    """
    self.game = game
    self.nnet = nnet
    self.args = args
    self.mcts = MCTS(game, nnet, args)

  def play(self, canonicalBoard, temp=1):
    """
    Simulate Play on Canonical Board

    Args:
      canonicalBoard: np.ndarray
        Canonical Board of size n x n [6x6 in this case]
      temp: Integer
        Signifies if game is in terminal state

    Returns:
      List of probabilities for all actions if temp is 0
      Best action based on max probability otherwise
    """
    for i in range(self.args.numMCTSSims):
      self.mcts.search(canonicalBoard)

    s = self.game.stringRepresentation(canonicalBoard)
    self.Nsa = self.mcts.getNsa()
    self.counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

    if temp == 0:
      bestAs = np.array(np.argwhere(self.counts == np.max(self.counts))).flatten()
      bestA = np.random.choice(bestAs)
      probs = [0] * len(self.counts)
      probs[bestA] = 1
      return probs

    self.counts = [x ** (1. / temp) for x in self.counts]
    self.counts_sum = float(sum(self.counts))
    probs = [x / self.counts_sum for x in self.counts]
    return np.argmax(probs)

  def getActionProb(self, canonicalBoard, temp=1):
    """
    Helper function to get probabilities associated with each action

    Args:
      canonicalBoard: np.ndarray
        Canonical Board of size n x n [6x6 in this case]
      temp: Integer
        Signifies if game is in terminal state

    Returns:
      action_probs: List
        Probability associated with corresponding action
    """
    action_probs = np.zeros((self.game.getActionSize()))
    best_action = self.play(canonicalBoard)
    action_probs[best_action] = 1

    return action_probs

set_seed(seed=SEED)
game = OthelloGame(6)
rp = RandomPlayer(game).play  # All players
num_games = 20  # Games
n1 = NNet(game)  # nnet players
n1.load_checkpoint(folder=path, filename=mcts_model_save_name)
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})

## Uncomment below to check your agent!
print('\n******MCTS player versus random player******')
mcts1 = MonteCarloTreeSearchBasedPlayer(game, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
arena = Arena.Arena(n1p, rp, game, display=OthelloGame.display)
MCTS_result = arena.playGames(num_games, verbose=False)
print(f"\n\n{MCTS_result}")
print(f"\nNumber of games won by player1 = {MCTS_result[0]}, "
      f"number of games won by player2 = {MCTS_result[1]}, out of {num_games} games")
win_rate_player1 = MCTS_result[0]/num_games
print(f"\nWin rate for player1 over {num_games} games: {round(win_rate_player1*100, 1)}%")