def cross_entropy_loss(x, labels):

  x_of_labels = torch.zeros(len(labels))
  for i, label in enumerate(labels):
    # 1. prediction for each class corresponding to the label
    x_of_labels[i] = x[i, label]

  # 2. loss vector for the batch
  losses = -x_of_labels + torch.log(torch.sum(torch.exp(x), axis=1))

  # 3. Return the average of the loss vector
  avg_loss = losses.mean()

  return avg_loss

### Uncomment below to test your function
labels = torch.tensor([0,
                       1])
x = torch.tensor([[10.0, 1.0, -1.0, -20.0], # correctly classified
                  [10.0, 10.0, 100.0, -110.0]]) # Not correctly classified

our_loss = cross_entropy_loss(x, labels).item()

CE = nn.CrossEntropyLoss()
pytorch_loss = CE(x, labels).item()
print('Our CE loss: %0.8f, Pytorch CE loss: %0.8f'%(our_loss, pytorch_loss))