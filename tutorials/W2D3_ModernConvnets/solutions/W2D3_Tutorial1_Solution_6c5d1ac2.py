def convolution_math(in_channels, filter_size, out_channels):
  """
  Convolution math: Implement how parameters scale as a function of feature maps
  and filter size in convolution vs depthwise separable convolution.

  Args:
    in_channels : int
      Number of input channels
    filter_size : int
      Size of the filter
    out_channels : int
      Number of output channels

  Returns:
    None
  """
  # Calculate the number of parameters for regular convolution
  conv_parameters = in_channels * filter_size * filter_size * out_channels
  # Calculate the number of parameters for depthwise separable convolution
  depthwise_conv_parameters = in_channels * filter_size * filter_size + in_channels * out_channels

  print(f"Depthwise separable: {depthwise_conv_parameters} parameters")
  print(f"Regular convolution: {conv_parameters} parameters")

  return None



## Uncomment to test your function
convolution_math(in_channels=4, filter_size=3, out_channels=2)