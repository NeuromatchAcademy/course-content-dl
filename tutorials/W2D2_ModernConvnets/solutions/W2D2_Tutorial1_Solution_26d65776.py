def get_parameter_count(network):
  """
  Calculate the number of parameters used by the fully connected network.
  Hint: Casting the result of network.parameters() to a list may make it
        easier to work with

  Args:
      network: Network to calculate the parameters of

  Returns:
      param_count: The number of parameters in the network
  """

  # Get the network's parameters
  parameters = network.parameters()

  param_count = 0
  # Loop over all layers
  for layer in parameters:
    param_count += torch.numel(layer)

  return param_count


# Initialize networks
fccnet = FullyConnectedNet()
convnet = ConvNet()
# Apply above defined function to both networks
## Uncomment to test your fuimction
print('FCCN parameter count: ' + str(get_parameter_count(fccnet)))
print('ConvNet parameter count: ' + str(get_parameter_count(convnet)))