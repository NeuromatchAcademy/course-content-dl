class ValueNetwork(NeuralNet):
  def __init__(self, game):
    self.nnet = OthelloNNet(game, args)
    self.board_x, self.board_y = game.getBoardSize()
    self.action_size = game.getActionSize()

    if args.cuda:
      self.nnet.cuda()

  def train(self, games):
    """
    examples: list of examples, each example is of form (board, pi, v)
    """
    optimizer = optim.Adam(self.nnet.parameters())
    for examples in games:
      for epoch in range(args.epochs):
        print('EPOCH ::: ' + str(epoch + 1))
        self.nnet.train()
        v_losses = []   # to store the losses per epoch
        batch_count = int(len(examples) / args.batch_size)  # len(examples)=200, batch-size=64, batch_count=3
        t = tqdm(range(batch_count), desc='Training Value Network')
        for _ in t:
          sample_ids = np.random.randint(len(examples), size=args.batch_size)  # read the ground truth information from MCTS simulation using the loaded examples
          boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))  # length of boards, pis, vis = 64
          boards = torch.FloatTensor(np.array(boards).astype(np.float64))
          target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

          # predict
          if args.cuda: # to run on GPU if available
            boards, target_vs = boards.contiguous().cuda(), target_vs.contiguous().cuda()

          # compute output
          _, out_v = self.nnet(boards)
          l_v = self.loss_v(target_vs, out_v)  # total loss

          # record loss
          v_losses.append(l_v.item())
          t.set_postfix(Loss_v=l_v.item())

          # compute gradient and do SGD step
          optimizer.zero_grad()
          l_v.backward()
          optimizer.step()

  def predict(self, board):
    """# Mean squared error (MSE)
    board: np array with board
    """
    # timing
    start = time.time()

    # preparing input
    board = torch.FloatTensor(board.astype(np.float64))
    if args.cuda:
      board = board.contiguous().cuda()
    board = board.view(1, self.board_x, self.board_y)
    self.nnet.eval()
    with torch.no_grad():
        _, v = self.nnet(board)
    return v.data.cpu().numpy()[0]

  def loss_v(self, targets, outputs):
    # Mean squared error (MSE)
    return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

  def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(folder, filename)
    if not os.path.exists(folder):
      print("Checkpoint Directory does not exist! Making directory {}".format(folder))
      os.mkdir(folder)
    else:
      print("Checkpoint Directory exists! ")
    torch.save({'state_dict': self.nnet.state_dict(),}, filepath)
    print("Model saved! ")

  def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
      raise ("No model in path {}".format(filepath))
    map_location = None if args.cuda else 'cpu'
    checkpoint = torch.load(filepath, map_location=map_location)
    self.nnet.load_state_dict(checkpoint['state_dict'])