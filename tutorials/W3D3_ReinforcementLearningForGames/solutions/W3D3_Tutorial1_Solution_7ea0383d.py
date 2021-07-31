class MonteCarloTreeSearchBasedPlayer():
  def __init__(self, game, nnet, args):
    self.game = game
    self.nnet = nnet
    self.args = args
    self.mcts = MCTS(game, nnet, args)

  def play(self, canonicalBoard, temp=1):
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
    action_probs = np.zeros((self.game.getActionSize()))
    best_action = self.play(canonicalBoard)
    action_probs[best_action] = 1

    return action_probs


# Load MCTS model from the repository
mcts_model_save_name = 'MCTS.pth.tar'
path = F"/content/nma_rl_games/alpha-zero/pretrained_models/models/"
game = OthelloGame(6)
rp = RandomPlayer(game).play  # all players
num_games = 20  # games
n1 = NNet(game)  # nnet players
n1.load_checkpoint(folder=path, filename=mcts_model_save_name)
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})

## Uncomment below to check your agent!
mcts1 = MonteCarloTreeSearchBasedPlayer(game, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
arena = Arena.Arena(n1p, rp, game, display=OthelloGame.display)
MCTS_result = arena.playGames(num_games, verbose=False)
print(f"\n\n{MCTS_result}")
print(f"\nNumber of games won by player1 = {MCTS_result[0]}, "
      f"number of games won by player2 = {MCTS_result[1]}, out of {num_games} games")
win_rate_player1 = MCTS_result[0]/num_games
print(f"\nWin rate for player1 over {num_games} games: {round(win_rate_player1*100, 1)}%")