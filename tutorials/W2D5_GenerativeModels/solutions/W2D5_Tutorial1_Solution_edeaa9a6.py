def gen_from_pPCA(noise_var, data_mean, pc_axes, pc_variance):
  """
  Generate samples from pPCA

  Args:
    noise_var: np.ndarray
      Sensor noise variance
    data_mean: np.ndarray
      Thermometer data mean
    pc_axes: np.ndarray
      Principal component axes
    pc_variance: np.ndarray
      The variance of the projection on the PC axes

  Returns:
    therm_data_sim: np.ndarray
      Generated (simulate, draw) `n_samples` from pPCA model
  """
  # We are matching this value to the thermometer data so the visualizations look similar
  n_samples = 1000

  # Randomly sample from z (latent space value)
  z = np.random.normal(0.0, np.sqrt(pc_variance), n_samples)

  # Sensor noise covariance matrix (∑)
  epsilon_cov = [[noise_var, 0.0], [0.0, noise_var]]

  # Data mean reshaped for the generation
  sim_mean = np.outer(data_mean, np.ones(n_samples))

  # Draw `n_samples` from `np.random.multivariate_normal`
  rand_eps = np.random.multivariate_normal([0.0, 0.0], epsilon_cov, n_samples)
  rand_eps = rand_eps.T

  # Generate (simulate, draw) `n_samples` from pPCA model
  therm_data_sim = sim_mean + np.outer(pc_axes, z) + rand_eps

  return therm_data_sim


# Add event to airtable
atform.add_event('Coding Exercise 2: pPCA')

## Uncomment to test your code
therm_data_sim = gen_from_pPCA(sensor_noise_var, therm_data_mean, pc_axes, pc_axes_variance)
with plt.xkcd():
  plot_gen_samples_ppca(therm1, therm2, therm_data_sim)