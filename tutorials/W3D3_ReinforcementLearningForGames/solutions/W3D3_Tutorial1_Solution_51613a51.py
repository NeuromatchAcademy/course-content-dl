class PolicyNetwork(NeuralNet):
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
        pi_losses = []

        batch_count = int(len(examples) / args.batch_size)

        t = tqdm(range(batch_count), desc='Training Policy Network')
        for _ in t:
          sample_ids = np.random.randint(len(examples), size=args.batch_size)
          boards, pis, _ = list(zip(*[examples[i] for i in sample_ids]))
          boards = torch.FloatTensor(np.array(boards).astype(np.float64))
          target_pis = torch.FloatTensor(np.array(pis))

          # predict
          if args.cuda:
            boards, target_pis = boards.contiguous().cuda(), target_pis.contiguous().cuda()

          # compute output
          out_pi, _ = self.nnet(boards)
          l_pi = self.loss_pi(target_pis, out_pi)

          # record loss
          pi_losses.append(l_pi.item())
          t.set_postfix(Loss_pi=l_pi.item())

          # compute gradient and do SGD step
          optimizer.zero_grad()
          l_pi.backward()
          optimizer.step()

  def predict(self, board):
    """
    board: np array with board
    """
    # timing
    start = time.time()

    # preparing input
    board = torch.FloatTensor(board.astype(np.float64))
    if args.cuda: board = board.contiguous().cuda()
    board = board.view(1, self.board_x, self.board_y)
    self.nnet.eval()
    with torch.no_grad():
      pi,_ = self.nnet(board)
    return torch.exp(pi).data.cpu().numpy()[0]

  def loss_pi(self, targets, outputs):
    # loss function. compute and return the negative log likelihood of targets!
    return -torch.sum(targets * outputs) / targets.size()[0]

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


game = OthelloGame(6)
## we use the same actor-critic network to output a policy
pnet = PolicyNetwork(game)
pnet.train(loaded_games)