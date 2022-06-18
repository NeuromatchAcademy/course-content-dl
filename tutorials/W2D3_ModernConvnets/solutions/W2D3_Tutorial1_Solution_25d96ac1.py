def get_parameter_count(network):
  """
  Calculate the number of parameters used by the fully connected/convolutional network.
  Hint: Casting the result of network.parameters() to a list may make it
        easier to work with

  Args:
    network: nn.module
      Network to calculate the parameters of fully connected/convolutional network

  Returns:
    param_count: int
      The number of parameters in the network
  """

  # Get the network's parameters
  parameters = network.parameters()

  param_count = 0
  # Loop over all layers
  for layer in parameters:
    param_count += torch.numel(layer)

  return param_count


# Add event to airtable
atform.add_event('Coding Exercise 1: Calculate number of parameters in FCNN vs ConvNet')

# Initialize networks
fccnet = FullyConnectedNet()
convnet = ConvNet()
## Apply the above defined function to both networks by uncommenting the following lines
print(f"FCCN parameter count: {get_parameter_count(fccnet)}")
print(f"ConvNet parameter count: {get_parameter_count(convnet)}")