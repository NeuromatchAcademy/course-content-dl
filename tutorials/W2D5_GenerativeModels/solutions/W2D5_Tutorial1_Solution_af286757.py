class LinearAutoEncoder(nn.Module):
  def __init__(self, K):
    super(LinearAutoEncoder, self).__init__()
    # encoder
    self.enc_lin = nn.Linear(my_dataset_dim, K)
    # decoder
    self.dec_lin = nn.Linear(K, my_dataset_dim)

  def encode(self, x):
    h = self.enc_lin(x)
    return h

  def decode(self, h):
    x_prime = self.dec_lin(h)
    return x_prime

  def forward(self, x):
    flat_x = x.view(x.size()[0], -1)
    h = self.encode(flat_x)
    return self.decode(h).view(x.size())


def train_autoencoder(autoencoder, dataset, epochs=20, batch_size=250):
  autoencoder.to(DEVICE)
  optim = torch.optim.Adam(autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)
  loss_fn = nn.MSELoss()
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      pin_memory=True, num_workers=2)
  mse_loss = torch.zeros(epochs * len(dataset) // batch_size, device=DEVICE)
  i = 0
  for epoch in trange(epochs, desc='Epoch'):
    for im_batch, _ in loader:
      im_batch = im_batch.to(DEVICE)
      optim.zero_grad()
      reconstruction = autoencoder(im_batch)
      # write the loss calculation
      loss = loss_fn(reconstruction.view(batch_size, -1),
                     target=im_batch.view(batch_size, -1))
      loss.backward()
      optim.step()

      mse_loss[i] = loss.detach()
      i += 1
  # After training completes, make sure the model is on CPU so we can easily
  # do more visualizations and demos.
  autoencoder.to('cpu')
  return mse_loss.cpu()


# Pick your own K
K = 20
set_seed(2021)
# Uncomment to test your code
lin_ae = LinearAutoEncoder(K)
lin_losses = train_autoencoder(lin_ae, my_dataset)
with plt.xkcd():
  plot_linear_ae(lin_losses)