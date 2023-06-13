class ValueNetwork(NeuralNet):

  def __init__(self, game):
    """
    Args:
      game: OthelloGame
        Instance of the OthelloGame class above
    """
    self.nnet = OthelloNNet(game, args)
    self.board_x, self.board_y = game.getBoardSize()
    self.action_size = game.getActionSize()
    self.nnet.to(args.device)

  def train(self, games):
    """
    Args:
      games: list
        List of examples with each example is of form (board, pi, v)
    """
    optimizer = optim.Adam(self.nnet.parameters())
    for examples in games:
      for epoch in range(args.epochs):
        print('EPOCH ::: ' + str(epoch + 1))
        self.nnet.train()
        v_losses = []   # To store the losses per epoch
        batch_count = int(len(examples) / args.batch_size)  # len(examples)=200, batch-size=64, batch_count=3
        t = tqdm(range(batch_count), desc='Training Value Network')
        for _ in t:
          sample_ids = np.random.randint(len(examples), size=args.batch_size)  # Read the ground truth information from MCTS simulation using the loaded examples
          boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))  # Length of boards, pis, vis = 64
          boards = torch.FloatTensor(np.array(boards).astype(np.float64))
          target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

          # Predict
          # To run on GPU if available
          boards, target_vs = boards.contiguous().to(args.device), target_vs.contiguous().to(args.device)

          # Compute output
          _, out_v = self.nnet(boards)
          l_v = self.loss_v(target_vs, out_v)  # Total loss

          # Record loss
          v_losses.append(l_v.item())
          t.set_postfix(Loss_v=l_v.item())

          # Compute gradient and do SGD step
          optimizer.zero_grad()
          l_v.backward()
          optimizer.step()

  def predict(self, board):
    """
    Args:
      board: np.ndarray
        Board of size n x n [6x6 in this case]

    Returns:
      v: OthelloNet instance
        Data of the OthelloNet class instance above;
    """
    # Timing
    start = time.time()

    # Preparing input
    board = torch.FloatTensor(board.astype(np.float64))
    board = board.contiguous().to(args.device)
    board = board.view(1, self.board_x, self.board_y)
    self.nnet.eval()
    with torch.no_grad():
        _, v = self.nnet(board)
    return v.data.cpu().numpy()[0]

  def loss_v(self, targets, outputs):
    """
    Args:
      targets: np.ndarray
        Ground Truth variables corresponding to input
      outputs: np.ndarray
        Predictions of Network

    Returns:
      MSE Loss averaged across the whole dataset
    """
    # Mean squared error (MSE)
    return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

  def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
    save_model_checkpoint(folder, filename, self.nnet)

  def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
    load_model_checkpoint(folder, filename, self.nnet, args.device)