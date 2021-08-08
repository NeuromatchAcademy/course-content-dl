def convolution_math(in_channels, filter_size, out_channels):
  """
  Convolution math: Implement how parameters scale as a function of feature maps
  and filter size in convolution vs depthwise separable convolution.

  Args:
    in_channels : number of input channels
    filter_size : size of the filter
    out_channels : number of output channels
  """
  # calculate the number of parameters for regular convolution
  conv_parameters = in_channels * filter_size * filter_size * out_channels
  # calculate the number of parameters for depthwise separable convolution
  depthwise_conv_parameters = in_channels * filter_size * filter_size + in_channels * out_channels

  print(f"Depthwise separable: {depthwise_conv_parameters} parameters")
  print(f"Regular convolution: {conv_parameters} parameters")

  return None


# add event to airtable
atform.add_event('Coding Exercise 6.1: Calculation of parameters')

## Uncomment to test your function
convolution_math(in_channels=4, filter_size=3, out_channels=2)