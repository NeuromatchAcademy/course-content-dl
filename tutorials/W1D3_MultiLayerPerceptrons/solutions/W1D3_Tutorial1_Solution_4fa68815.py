
class Net(nn.Module):
    def __init__(self, actv, num_inputs, hidden_units, num_outputs):
        super(Net, self).__init__()

        # Assign activation function (exec allows us to assign function from string)
        exec('self.actv = nn.%s'%actv)

        # Initialize layers of MLP
        self.layers = nn.ModuleList()

        # Loop over layers and create each one
        for i in range(len(hidden_units)):
          next_num_inputs = hidden_units[i]
          self.layers += [nn.Linear(num_inputs, next_num_inputs)]
          num_inputs = next_num_inputs

        # Create final layer
        self.out = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):

        # Flatten inputs to 2D (if more than that)
        x = x.view(x.shape[0], -1)

        # Get activations of each layer
        for layer in self.layers:
          x = self.actv(layer(x))

        # Get outputs
        x = self.out(x)

        return x

### Uncomment below to create network and test it on input
net = Net(actv='LeakyReLU(0.1)',
    num_inputs = 2,
    hidden_units = [100, 10, 5],
    num_outputs = 1)

input = torch.zeros((100, 2))
y = net(input)
print(f'The output shape is {y.shape} for an input of shape {input.shape}')