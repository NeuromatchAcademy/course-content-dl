def cross_entropy_loss(x, labels):
  # x is the model predictions we'd like to evaluate using lables
  x_of_labels = torch.zeros(len(labels))
  # 1. prediction for each class corresponding to the label
  for i, label in enumerate(labels):
    x_of_labels[i] = x[i, label]
  # 2. loss vector for the batch
  losses = -x_of_labels + torch.log(torch.sum(torch.exp(x), axis=1))
  # 3. Return the average of the loss vector
  avg_loss = losses.mean()

  return avg_loss

# add event to airtable
atform.add_event('Coding Exercise 3.1: Implement Batch Cross Entropy Loss')


labels = torch.tensor([0, 1])
x = torch.tensor([[10.0, 1.0, -1.0, -20.0],  # correctly classified
                  [10.0, 10.0, 2.0, -10.0]])  # Not correctly classified
CE = nn.CrossEntropyLoss()
pytorch_loss = CE(x, labels).item()
## Uncomment below to test your function
our_loss = cross_entropy_loss(x, labels).item()
print(f'Our CE loss: {our_loss:0.8f}, Pytorch CE loss: {pytorch_loss:0.8f}')
print(f'Difference: {np.abs(our_loss - pytorch_loss):0.8f}')