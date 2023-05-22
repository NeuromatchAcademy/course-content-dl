class MonteCarlo():
  """
  Implementation of Monte Carlo Algorithm
  """

  def __init__(self, game, nnet, args):
    """
    Initialize Monte Carlo Parameters

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

    self.Ps = {}  # Stores initial policy (returned by neural net)
    self.Es = {}  # Stores game.getGameEnded ended for board s

  # Call this rollout
  def simulate(self, canonicalBoard):
    """
    Helper function to simulate one Monte Carlo rollout

    Args:
      canonicalBoard: np.ndarray
        Canonical Board of size n x n [6x6 in this case]

    Returns:
      temp_v:
        Terminal State
    """
    s = self.game.stringRepresentation(canonicalBoard)
    init_start_state = s
    temp_v = 0
    isfirstAction = None
    current_player = -1  # opponent's turn (the agent has already taken an action before the simulation)
    self.Ps[s], _ = self.nnet.predict(canonicalBoard)

    for i in range(self.args.maxDepth):  # maxDepth

      if s not in self.Es:
        self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
      if self.Es[s] != 0:
        # Terminal state
        temp_v = -self.Es[s] * current_player
        break

      self.Ps[s], v = self.nnet.predict(canonicalBoard)
      valids = self.game.getValidMoves(canonicalBoard, 1)
      self.Ps[s] = self.Ps[s] * valids  # Masking invalid moves
      sum_Ps_s = np.sum(self.Ps[s])

      if sum_Ps_s > 0:
        self.Ps[s] /= sum_Ps_s  # Renormalize
      else:
        # If all valid moves were masked make all valid moves equally probable
        # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
        # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
        log.error("All valid moves were masked, doing a workaround.")
        self.Ps[s] = self.Ps[s] + valids
        self.Ps[s] /= np.sum(self.Ps[s])

      # Take a random action
      a = np.random.choice(self.game.getActionSize(), p=self.Ps[s])
      # Find the next state and the next player
      next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
      canonicalBoard = self.game.getCanonicalForm(next_s, next_player)
      s = self.game.stringRepresentation(next_s)
      current_player *= -1
      # Initial policy
      self.Ps[s], v = self.nnet.predict(canonicalBoard)
      temp_v = v

    return temp_v
