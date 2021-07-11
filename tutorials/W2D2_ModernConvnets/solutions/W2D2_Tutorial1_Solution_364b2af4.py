def convolution_math(in_features, filter_size, out_features):
  """
  Convolution math: Implement how parameters scale as a function of feature maps
  and filter size in convolution vs depthwise separable convolution.

  Args:
    in_features:  number of input features
    filter_size:  size of the filter
    out_features: number of output features
  """
  # calculate the number of parameters for regular convolution
  conv_parameters = in_features * filter_size * filter_size * out_features
  # calculate the number of parameters for depthwise separable convolution
  depthwise_conv_parameters = in_features * filter_size * filter_size + in_features * out_features

  print('Depthwise separable: {} parameters'.format(depthwise_conv_parameters))
  print('Regular convolution: {} parameters'.format(conv_parameters))

  return None


## Uncomment to test your function
convolution_math(in_features=4, filter_size=3, out_features=2)