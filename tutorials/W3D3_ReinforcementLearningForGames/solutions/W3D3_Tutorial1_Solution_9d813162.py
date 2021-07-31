class MonteCarloBasedPlayer():
  def __init__(self, game, nnet, args):
    self.game = game
    self.nnet = nnet
    self.args = args
    self.mc = MonteCarlo(game, nnet, args)
    self.K = self.args.mc_topk

  def play(self, canonicalBoard):
    self.qsa = []
    s = self.game.stringRepresentation(canonicalBoard)
    Ps, v = self.nnet.predict(canonicalBoard)
    valids = self.game.getValidMoves(canonicalBoard, 1)
    Ps = Ps * valids  # masking invalid moves
    sum_Ps_s = np.sum(Ps)

    if sum_Ps_s > 0:
      Ps /= sum_Ps_s  # renormalize
    else:
      # if all valid moves were masked make all valid moves equally probable
      # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
      # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
      log = logging.getLogger(__name__)
      log.error("All valid moves were masked, doing a workaround.")
      Ps = Ps + valids
      Ps /= np.sum(Ps)

    num_valid_actions = np.shape(np.nonzero(Ps))[1]

    if num_valid_actions < self.K:
      top_k_actions = np.argpartition(Ps,-num_valid_actions)[-num_valid_actions:]
    else:
      top_k_actions = np.argpartition(Ps,-self.K)[-self.K:]  # to get actions that belongs to top k prob

    for action in top_k_actions:
      next_s, next_player = self.game.getNextState(canonicalBoard, 1, action)
      next_s = self.game.getCanonicalForm(next_s, next_player)

      values = []

      # do some rollouts
      for rollout in range(self.args.numMCsims):
        value = self.mc.simulate(canonicalBoard)
        values.append(value)

      # average out values
      avg_value = np.mean(values)
      self.qsa.append((avg_value, action))

    self.qsa.sort(key=lambda a: a[0])
    self.qsa.reverse()
    best_action = self.qsa[0][1]
    return best_action

  def getActionProb(self, canonicalBoard, temp=1):
    if self.game.getGameEnded(canonicalBoard, 1) != 0:
      return np.zeros((self.game.getActionSize()))

    else:
      action_probs = np.zeros((self.game.getActionSize()))
      best_action = self.play(canonicalBoard)
      action_probs[best_action] = 1

    return action_probs


set_seed(seed=SEED)
game = OthelloGame(6)
rp = RandomPlayer(game).play  # all players
num_games = 20  # Feel free to change this number

n1 = NNet(game)  # nNet players
n1.load_checkpoint(folder=path, filename=mc_model_save_name)
args1 = dotdict({'numMCsims': 10, 'maxRollouts':5, 'maxDepth':5, 'mc_topk': 3})

## Uncomment below to check Monte Carlo agent!
mc1 = MonteCarloBasedPlayer(game, n1, args1)
n1p = lambda x: np.argmax(mc1.getActionProb(x))
arena = Arena.Arena(n1p, rp, game, display=OthelloGame.display)
MC_result = arena.playGames(num_games, verbose=False)
print(f"\n\n{MC_result}")
print(f"\nNumber of games won by player1 = {MC_result[0]}, "
      f"number of games won by player2 = {MC_result[1]}, out of {num_games} games")
win_rate_player1 = result[0]/num_games
print(f"\nWin rate for player1 over {num_games} games: {round(win_rate_player1*100, 1)}%")