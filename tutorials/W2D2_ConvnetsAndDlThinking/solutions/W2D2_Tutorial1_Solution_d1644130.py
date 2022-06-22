def calculate_output_shape(image_shape, kernel_shape):
  """
  Helper function to calculate output shape

  Args:
    image_shape: tuple
      Image shape
    kernel_shape: tuple
      Kernel shape

  Returns:
    output_height: int
      Output Height
    output_width: int
      Output Width
  """
  image_height, image_width = image_shape
  kernel_height, kernel_width = kernel_shape
  output_height = image_height - kernel_height + 1
  output_width = image_width - kernel_width + 1
  return output_height, output_width


# Add event to airtable
atform.add_event('Coding Exercise 2.2: Convolution Output Size')

# Here we check if your function works correcly by applying it to different image
# and kernel shapes
check_shape_function(calculate_output_shape, image_shape=(3, 3), kernel_shape=(2, 2))
check_shape_function(calculate_output_shape, image_shape=(3, 4), kernel_shape=(2, 3))
check_shape_function(calculate_output_shape, image_shape=(5, 5), kernel_shape=(5, 5))
check_shape_function(calculate_output_shape, image_shape=(10, 20), kernel_shape=(3, 2))
check_shape_function(calculate_output_shape, image_shape=(100, 200), kernel_shape=(40, 30))