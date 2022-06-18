# Define the Siamese Network
class ConvSiameseNet(nn.Module):
  """
  Convolutional Siamese Network from "Siamese Neural Networks for One-shot Image Recognition"
  Paper can be found at http://www.cs.toronto.edu/~rsalakhu/papers/oneshot1.pdf

  Structure of the network is as follows:
  nn.Conv2d(1, 64, 10) + pool(F.relu(self.conv1(x))) # First Convolutional + Pooling Block
  nn.Conv2d(64, 128, 7) + pool(F.relu(self.conv2(x))) # Second Convolutional + Pooling Block
  nn.Conv2d(128, 128, 4) + pool(F.relu(self.conv3(x))) # Third Convolutional + Pooling Block
  nn.Conv2d(128, 256, 4) + F.relu(self.conv4(x)) # Fourth Convolutional Layer
  nn.MaxPool2d(2, 2) # Pooling Block
  nn.Linear(256*6*6, 4096) # First Fully Connected Layer
  nn.Linear(4096, 1) # Second Fully Connected Layer
  """

  def __init__(self):
    """
    Initialize convolutional Siamese network parameters

    Args:
      None

    Returns:
      Nothing
    """
    super().__init__()
    self.conv1 = nn.Conv2d(1, 64, 10)
    self.conv2 = nn.Conv2d(64, 128, 7)
    self.conv3 = nn.Conv2d(128, 128, 4)
    self.conv4 = nn.Conv2d(128, 256, 4)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(256*6*6, 4096)
    self.fc2 = nn.Linear(4096, 1)

  def model(self, x):
    """
    Defines model structure and flow

    Args:
      x: Dataloader instance
        Input Dataset

    Returns:
      x: torch.tensor
        Output of first fully connected layer
    """
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    x = F.relu(self.conv4(x))
    x = torch.flatten(x, 1)
    x = torch.sigmoid(self.fc1(x))
    return x

  def forward(self, x1, x2):
    """
    Calculates L1 distance between model pass of sample 1 and sample 2

    Args:
      x1: torch.tensor
        Sample 1
      x2: torch.tensor
        Sample 2

    Returns:
      Output from final fully connected layer on recieving L1 distance as input
    """
    x1_fv = self.model(x1)
    x2_fv = self.model(x2)
    # Calculate L1 distance (as l1_distance) between x1_fv and x2_fv
    l1_distance = torch.abs(x1_fv - x2_fv)

    return self.fc2(l1_distance)