def exercise_1(η=100.0, epochs=250 , γ=1e-12):
  """Training a LNN

  Args:
  η (float): learning rate (default 100.0)
  epochs (int): number of epochs (default 250)
  γ (float): initialization scale (default 1e-12)

  """
  n_hidden = [30]

  dim_input = tree_labels.shape[1]
  dim_output = tree_features.shape[1]

  deep_model = VariableDepthWidth(in_dim=dim_input,
                                     out_dim=dim_output,
                                     hid_dims=n_hidden,
                                     gamma=γ)

  # convert (cast) data from np.ndarray to torch.Tensor
  input_tensor = torch.tensor(tree_labels).float()
  output_tensor = torch.tensor(tree_features).float()

  training_losses = train(deep_model,
                          input_tensor,
                          output_tensor,
                          n_epochs=epochs,
                          lr=η)

  plot_loss(training_losses)


# # Uncomment and run
with plt.xkcd():
  exercise_1()