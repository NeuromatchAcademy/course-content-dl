def sample_minibatch(input_data, target_data, num_points=100):
  """
  Sample a minibatch of size num_point from the provided input-target data

  Args:
    input_data: Tensor
      Multi-dimensional tensor containing the input data
    target_data: Tensor
      1D tensor containing the class labels
    num_points: Integer
      Number of elements to be included in minibatch with default=100

  Returns:
    batch_inputs: Tensor
      Minibatch inputs
    batch_targets: Tensor
      Minibatch targets
  """
  # Sample a collection of IID indices from the existing data
  batch_indices = np.random.choice(len(input_data), num_points)
  # Use batch_indices to extract entries from the input and target data tensors
  batch_inputs = input_data[batch_indices, :]
  batch_targets = target_data[batch_indices]

  return batch_inputs, batch_targets

# add event to airtable
atform.add_event('Coding Exercise 6: Implement minibatch sampling')


## Uncomment to test your function
x_batch, y_batch = sample_minibatch(X, y, num_points=100)
print(f"The input shape is {x_batch.shape} and the target shape is: {y_batch.shape}")