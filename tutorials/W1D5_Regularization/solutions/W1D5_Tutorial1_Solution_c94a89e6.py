def early_stopping_main(args, model, train_loader, val_loader):
  device = args['device']
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
    trained_model = train(args, model, train_loader, optimizer)

    # calculate training accuracy
    train_acc = test(trained_model, train_loader, device=device)

    # calculate validation accuracy
    val_acc = test(trained_model, val_loader, device=device)

    if (val_acc > best_acc):
      best_acc = val_acc
      best_epoch = epoch
      best_model = copy.deepcopy(trained_model)
      wait = 0
    else:
      wait += 1

    if (wait > patience):
      print(f'early stopped on epoch: {epoch}')
      break

    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

  return val_acc_list, train_acc_list, best_model, best_epoch


# Set the arguments
args = {
    'epochs': 200,
    'lr': 5e-4,
    'momentum': 0.99,
    'device': DEVICE
}

# Initialize the model
set_seed(seed=SEED)
model = AnimalNet()

## Uncomment to test
val_acc_earlystop, train_acc_earlystop, best_model, best_epoch = early_stopping_main(args, model, train_loader, val_loader)
print(f'Maximum Validation Accuracy is reached at epoch: {best_epoch:2d}')
with plt.xkcd():
  early_stop_plot(train_acc_earlystop, val_acc_earlystop, best_epoch)