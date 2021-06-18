def get_fcnn_parameter_count() -> int:
    """
    Calculate the number of parameters used by the fully connected network.
    Hint: Casting the result of fc_net.parameters() to a list may make it
          easier to work with

    Returns:
        param_count: The number of parameters in the network
    """

    fc_net = FullyConnectedNet()

    fc_net_parameters = fc_net.parameters()
    param_count = 0
    for layer in fc_net_parameters:
        current_layer_params = None
        for dimension in layer.shape:
            if current_layer_params is None:
                current_layer_params = dimension
            else:
                current_layer_params *= dimension
        param_count += current_layer_params

    # Alternatively, there's a convenient torch function to count the number of items in a tensor:
    fc_net_parameters = fc_net.parameters()

    param_count = 0
    for layer in fc_net_parameters:
        param_count += torch.numel(layer)

    return param_count

### Uncomment below to test your function
print(get_fcnn_parameter_count())