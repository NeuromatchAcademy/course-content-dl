def get_cnn_parameter_count() -> int:
  """
  Calculate the number of parameters used by the convolutional network.
  Hint: Casting the result of cnn_net.parameters() to a list may make it
        easier to work with

  Returns:
      param_count: The number of parameters in the network
  """

  convnet = ConvNet()
  convnet_parameters = list(convnet.parameters())

  param_count = 0
  for layer in convnet_parameters:
    param_count += torch.numel(layer)

  return param_count

### Uncomment below to test your function
print(get_cnn_parameter_count())