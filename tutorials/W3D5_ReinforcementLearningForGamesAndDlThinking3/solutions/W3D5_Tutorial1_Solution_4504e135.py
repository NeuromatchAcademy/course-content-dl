class PolicyNetwork(NeuralNet):

  def __init__(self, game):
    """
    Args:
      game: OthelloGame
        Instance of the OthelloGame class
    """
    self.nnet = OthelloNNet(game, args)
    self.board_x, self.board_y = game.getBoardSize()
    self.action_size = game.getActionSize()
    self.nnet.to(args.device)

  def train(self, games):
    """
    Args:
      games: list
        List of examples where each example is of form (board, pi, v)
    """
    optimizer = optim.Adam(self.nnet.parameters())

    for examples in games:
      for epoch in range(args.epochs):
        print('EPOCH ::: ' + str(epoch + 1))
        self.nnet.train()
        pi_losses = []

        batch_count = int(len(examples) / args.batch_size)

        t = tqdm(range(batch_count), desc='Training Policy Network')
        for _ in t:
          sample_ids = np.random.randint(len(examples), size=args.batch_size)
          boards, pis, _ = list(zip(*[examples[i] for i in sample_ids]))
          boards = torch.FloatTensor(np.array(boards).astype(np.float64))
          target_pis = torch.FloatTensor(np.array(pis))

          # Predict
          boards, target_pis = boards.contiguous().to(args.device), target_pis.contiguous().to(args.device)

          # Compute output
          out_pi, _ = self.nnet(boards)
          l_pi = self.loss_pi(target_pis, out_pi)

          # Record loss
          pi_losses.append(l_pi.item())
          t.set_postfix(Loss_pi=l_pi.item())

          # Compute gradient and do SGD step
          optimizer.zero_grad()
          l_pi.backward()
          optimizer.step()

  def predict(self, board):
    """
    Args:
      board: np.ndarray
        Board of size n x n [6x6 in this case]

    Returns:
      Data from the OthelloNet instance
    """
    # Timing
    start = time.time()

    # Preparing input
    board = torch.FloatTensor(board.astype(np.float64))
    board = board.contiguous().to(args.device)
    board = board.view(1, self.board_x, self.board_y)
    self.nnet.eval()
    with torch.no_grad():
      pi,_ = self.nnet(board)
    return torch.exp(pi).data.cpu().numpy()[0]

  def loss_pi(self, targets, outputs):
    """
    Calculates Negative Log Likelihood(NLL) of Targets

    Args:
      targets: np.ndarray
        Ground Truth variables corresponding to input
      outputs: np.ndarray
        Predictions of Network

    Returns:
      Negative Log Likelihood calculated as: When training a model, we aspire to
      find the minima of a loss function given a set of parameters (in a neural
      network, these are the weights and biases).
      Sum the loss function to all the correct classes. So, whenever the network
      assigns high confidence at the correct class, the NLL is low, but when the
      network assigns low confidence at the correct class, the NLL is high.
    """
    ## For more information, here is a reference that connects the expression to
    # the neg-log-prob: https://gombru.github.io/2018/05/23/cross_entropy_loss/
    return -torch.sum(targets * outputs) / targets.size()[0]

  def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
    save_model_checkpoint(folder, filename, self.nnet)

  def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
    load_model_checkpoint(folder, filename, self.nnet, args.device)