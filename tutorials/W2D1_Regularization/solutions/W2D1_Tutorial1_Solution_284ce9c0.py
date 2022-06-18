def early_stopping_main(args, model, train_loader, val_loader):
  """
  Function to simulate early stopping

  Args:
    args: dictionary
      Dictionary with epochs: 200, lr: 5e-3, momentum: 0.9, device: DEVICE
    model: nn.module
      Neural network instance
    train_loader: torch.loader
      Train dataset
    val_loader: torch.loader
      Validation set

  Returns:
    val_acc_list: list
      Val accuracy log until early stop point
    train_acc_list: list
      Training accuracy log until early stop point
    best_model: nn.module
      Model performing best with early stopping
    best_epoch: int
      Epoch at which early stopping occurs
  """

  device = args['device']
  model = model.to(device)
  optimizer = optim.SGD(model.parameters(),
                        lr=args['lr'],
                        momentum=args['momentum'])

  best_acc = 0.0
  best_epoch = 0

  # Number of successive epochs that you want to wait before stopping training process
  patience = 20

  # Keeps track of number of epochs during which the val_acc was less than best_acc
  wait = 0

  val_acc_list, train_acc_list = [], []
  for epoch in tqdm(range(args['epochs'])):

    # Train the model
    trained_model = train(args, model, train_loader, optimizer)

    # Calculate training accuracy
    train_acc = test(trained_model, train_loader, device=device)

    # Calculate validation accuracy
    val_acc = test(trained_model, val_loader, device=device)

    if (val_acc > best_acc):
      best_acc = val_acc
      best_epoch = epoch
      best_model = copy.deepcopy(trained_model)
      wait = 0
    else:
      wait += 1

    if (wait > patience):
      print(f'Early stopped on epoch: {epoch}')
      break

    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

  return val_acc_list, train_acc_list, best_model, best_epoch


# Add event to airtable
atform.add_event('Coding Exercise 4: Early Stopping')

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