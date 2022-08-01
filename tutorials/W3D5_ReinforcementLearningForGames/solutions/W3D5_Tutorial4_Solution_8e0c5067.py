class MCTS():
  """
  This class handles MCTS (Monte Carlo Tree Search).
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
    self.Qsa = {}  # Stores Q values for s,a (as defined in the paper)
    self.Nsa = {}  # Stores #times edge s,a was visited
    self.Ns = {}  # Stores #times board s was visited
    self.Ps = {}  # Stores initial policy (returned by neural net)
    self.Es = {}  # Stores game.getGameEnded ended for board s
    self.Vs = {}  # Stores game.getValidMoves for board s

  def search(self, canonicalBoard):
    """
    This function performs one iteration of MCTS. It is recursively called
    till a leaf node is found. The action chosen at each node is one that
    has the maximum upper confidence bound as in the paper.
    Once a leaf node is found, the neural network is called to return an
    initial policy P and a value v for the state. This value is propagated
    up the search path. In case the leaf node is a terminal state, the
    outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
    updated.
    NOTE: the return values are the negative of the value of the current
    state. This is done since v is in [-1,1] and if v is the value of a
    state for the current player, then its value is -v for the other player.

    Args:
      canonicalBoard: np.ndarray
        Canonical Board of size n x n [6x6 in this case]

    Returns:
        v: Float
          The negative of the value of the current canonicalBoard
    """
    s = self.game.stringRepresentation(canonicalBoard)

    if s not in self.Es:
      self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
    if self.Es[s] != 0:
      # Terminal node
      return -self.Es[s]

    if s not in self.Ps:
      # Leaf node
      self.Ps[s], v = self.nnet.predict(canonicalBoard)
      valids = self.game.getValidMoves(canonicalBoard, 1)
      self.Ps[s] = self.Ps[s] * valids  # Masking invalid moves
      sum_Ps_s = np.sum(self.Ps[s])
      if sum_Ps_s > 0:
        self.Ps[s] /= sum_Ps_s  # Renormalize
      else:
        # If all valid moves were masked make all valid moves equally probable
        # NB! All valid moves may be masked if either your NNet architecture is
        # insufficient or you've get overfitting or something else.
        # If you have got dozens or hundreds of these messages you should
        # pay attention to your NNet and/or training process.
        log = logging.getLogger(__name__)
        log.error("All valid moves were masked, doing a workaround.")
        self.Ps[s] = self.Ps[s] + valids
        self.Ps[s] /= np.sum(self.Ps[s])

      self.Vs[s] = valids
      self.Ns[s] = 0

      return -v

    valids = self.Vs[s]
    cur_best = -float('inf')
    best_act = -1

    # Pick the action with the highest upper confidence bound
    for a in range(self.game.getActionSize()):
      if valids[a]:
        if (s, a) in self.Qsa:
          u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
        else:
          u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)

        if u > cur_best:
          cur_best = u
          best_act = a

    a = best_act
    next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
    next_s = self.game.getCanonicalForm(next_s, next_player)

    v = self.search(next_s)

    if (s, a) in self.Qsa:
      self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
      self.Nsa[(s, a)] += 1

    else:
      self.Qsa[(s, a)] = v
      self.Nsa[(s, a)] = 1

    self.Ns[s] += 1
    return -v

  def getNsa(self):
    return self.Nsa