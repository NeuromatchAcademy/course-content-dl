def early_stopping_main(args, model, train_loader, val_loader, test_data):

  use_cuda = not args['no_cuda'] and torch.cuda.is_available()
  device = torch.device('cuda' if use_cuda else 'cpu')

  model = model.to(device)
  optimizer = optim.SGD(model.parameters(),
                        lr=args['lr'],
                        momentum=args['momentum'])

  best_acc = 0.0
  best_epoch = 0

  # Number of successive epochs that you want to wait before stopping training process
  patience = 20

  # Keps track of number of epochs during which the val_acc was less than best_acc
  wait = 0

  val_acc_list, train_acc_list = [], []
  for epoch in tqdm(range(args['epochs'])):
    # train the model
    train(args, model, device, train_loader, optimizer, epoch)
    # calculate training accuracy
    train_acc = test(model, device, train_loader, 'Train')
    # calculate validation accuracy
    val_acc = test(model, device, val_loader, 'Val')
    if (val_acc > best_acc):
      best_acc = val_acc
      best_epoch = epoch
      best_model = copy.deepcopy(model)
      wait = 0
    else:
      wait += 1
    if (wait > patience):
      print(f'early stopped on epoch:{epoch}')
      break
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

  return val_acc_list, train_acc_list, best_model, best_epoch