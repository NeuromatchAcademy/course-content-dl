class Net(nn.Module):
  def __init__(self, actv, input_feature_num, hidden_unit_nums, output_feature_num):
    super(Net, self).__init__()

    # Assign activation function (eval allows us to instantiate object from string)
    self.actv = eval('nn.%s'%actv)

    # save the input size for later
    self.input_feature_num = input_feature_num

    # Initialize layers of MLP
    self.layers = nn.Sequential()

    # Loop over layers and create each one
    for i in range(len(hidden_unit_nums)):
      # assign the current layer output feature numbers from hidden layer list
      next_input_feature_num = hidden_unit_nums[i]
      # use nn.Linear to define the layer
      layer = nn.Linear(input_feature_num, next_input_feature_num)
      # append it to the model with a name
      self.layers.add_module('Linear%d'%i, layer)
      # assign next layer input using current layer output
      input_feature_num = next_input_feature_num

    # Create final layer
    self.out = nn.Linear(input_feature_num, output_feature_num)

  def forward(self, x):
    # reshape inputs to (batch_size, input_feature_num)
    # just in case the input vector is not 2D, like an image!
    x = x.view(-1, self.input_feature_num)
    # get each layer and run it on previous output and apply the activation function
    for layer in self.layers:
      x = self.actv(layer(x))

    # Get outputs
    x = self.out(x)

    return x


input = torch.zeros((100, 2))
## Uncomment below to create network and test it on input
net = Net(actv='LeakyReLU(0.1)', input_feature_num = 2,
          hidden_unit_nums = [100, 10, 5], output_feature_num = 1)
y = net(input)
print(f'The output shape is {y.shape} for an input of shape {input.shape}')