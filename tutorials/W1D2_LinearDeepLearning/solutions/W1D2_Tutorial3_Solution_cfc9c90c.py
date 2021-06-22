class VariableDepthWidthExercise(nn.Module):
  def __init__(self, in_dim, out_dim, hid_dims=[], gamma=1e-12):
    """Variable depth linear network

    Args:
      in_dim (int): input dimension
      out_dim (int): ouput dimension
      hid_dims (list): a list, containing the number of neurons in each hidden layer
        default is empty list (`[]`) for linear regression.
        example: For 2 hidden layers, first with 5 and second with 7 neurons,
                 we use: `hid_dims = [5, 7]`
    """
    super().__init__()
    assert isinstance(in_dim, int)
    assert isinstance(out_dim, int)
    assert isinstance(hid_dims, list)
    n_hidden_layers = len(hid_dims)  # number of hidden layers
    layers = OrderedDict()

    if n_hidden_layers == 0:  # linear regression
      layers["map"] = nn.Linear(in_dim, out_dim, bias=False)

    else:  # shallow and deep linear neural net
      layers["in->"] = nn.Linear(in_dim, hid_dims[0], bias=False)

      for i in range(n_hidden_layers-1):  # creating hidden layers
        layers["hid {}".format(i+1)] = nn.Linear(hid_dims[i],
                                                    hid_dims[i+1],
                                                    bias=False)

      layers["->out"] = nn.Linear(hid_dims[-1], out_dim, bias=False)

    for k in layers:  # re-initialization of the weights
      sigma = gamma / sqrt(layers[k].weight.shape[0] + layers[k].weight.shape[1])
      nn.init.normal_(layers[k].weight, std=sigma)

    self.layers = nn.Sequential(layers)

  def forward(self, input_tensor):
    """Forward pass
    """
    return self.layers(input_tensor)


# # Uncomment and run
print("Deep LNN:\n",
      VariableDepthWidthExercise(64, 100, [32, 16, 16, 32]))

print("\nLinear Regression model:\n",
      VariableDepthWidthExercise(64, 100,[]))