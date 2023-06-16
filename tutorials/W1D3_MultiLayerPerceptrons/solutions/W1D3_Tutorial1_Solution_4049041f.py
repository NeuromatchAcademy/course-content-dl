def cross_entropy_loss(x, labels):
  """
  Helper function to compute cross entropy loss

  Args:
    x: torch.tensor
      Model predictions we'd like to evaluate using labels
    labels: torch.tensor
      Ground truth

  Returns:
    avg_loss: float
      Average of the loss vector
  """
  x_of_labels = torch.zeros(len(labels))
  # 1. Prediction for each class corresponding to the label
  for i, label in enumerate(labels):
    x_of_labels[i] = x[i, label]
  # 2. Loss vector for the batch
  losses = -x_of_labels + torch.log(torch.sum(torch.exp(x), axis=1))
  # 3. Return the average of the loss vector
  avg_loss = losses.mean()

  return avg_loss



labels = torch.tensor([0, 1])
x = torch.tensor([[10.0, 1.0, -1.0, -20.0],  # Correctly classified
                  [10.0, 10.0, 2.0, -10.0]])  # Not correctly classified
CE = nn.CrossEntropyLoss()
pytorch_loss = CE(x, labels).item()
## Uncomment below to test your function
our_loss = cross_entropy_loss(x, labels).item()
print(f'Our CE loss: {our_loss:0.8f}, Pytorch CE loss: {pytorch_loss:0.8f}')
print(f'Difference: {np.abs(our_loss - pytorch_loss):0.8f}')