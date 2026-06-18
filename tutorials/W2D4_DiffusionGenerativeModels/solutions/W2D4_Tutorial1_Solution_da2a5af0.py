def reverse_diffusion_SDE_sampling_gmm(gmm, sampN=200, Lambda=5, nsteps=500):
  """ Using exact score function to simulate the reverse SDE to sample from distribution.

  Args:
    gmm:    Gaussian Mixture Model (defined above) — we use its .score() method for exact scores
    sampN:  Number of particles to simulate simultaneously
    Lambda: lambda in the diffusion coefficient g(t) = lambda^t
    nsteps: number of discrete time steps for the reverse process
  """
  sigmaT2 = sigma_t_square(1, Lambda)

  # xT: starting positions of all particles, drawn from the prior N(0, sigma_T^2 * I)
  # shape: (sampN, 2)
  #   sampN = number of particles being simulated in parallel
  #   2     = dimensionality of the data (each point is a 2D coordinate, x and y)
  #           NOTE: this 2 is NOT the number of Gaussian components in the GMM
  xT = sigmaT2 ** 0.5 * np.random.randn(sampN, 2)

  # x_traj_rev: stores particle positions at every time step
  # shape: (nsteps, sampN, 2)
  #   dim 0 (nsteps): time step index — x_traj_rev[0] is t=T (noise), x_traj_rev[-1] is t~0 (data)
  #   dim 1 (sampN):  which particle
  #   dim 2 (2):      x or y coordinate
  x_traj_rev = np.zeros((nsteps, sampN, 2))
  x_traj_rev[0] = xT   # all particles start from noise at t=T
  dt = 1 / nsteps

  for i in range(1, nsteps):
    t = 1 - i * dt  # time runs backward: T → 0

    # z_t ~ N(0, I): fresh Gaussian noise drawn at each step, shape: (sampN, 2)
    z_t = np.random.randn(sampN, 2)
    # (previously written as np.random.randn(*xT.shape), where *xT.shape unpacks
    # the tuple (sampN, 2) into separate arguments — equivalent to randn(sampN, 2))

    # diffuse_gmm analytically transports the GMM forward to time t
    gmm_t = diffuse_gmm(gmm, t, Lambda)
    # (adds noise variance sigma_t^2 to each component's covariance)
    # Because we know the GMM analytically, gmm_t.score() gives the EXACT score —
    # no estimation needed. In real diffusion models this would be a neural network.

    # s(x_t, t) = exact score nabla log p_t(x_t) for all sampN particles at once
    # score_xt shape: (sampN, 2)
    score_xt = gmm_t.score(x_traj_rev[i-1])

    # where g(t) = Lambda^t
    # one reverse step: x_{t-dt} = x_t + g(t)^2 * s(x_t,t)*dt + g(t)*sqrt(dt)*z_t
    x_traj_rev[i] = x_traj_rev[i-1] + z_t * (Lambda**t) * dt ** 0.5 + score_xt * dt * Lambda**(2*t)


# test your function
  return x_traj_rev


set_seed(42)
x_traj_rev = reverse_diffusion_SDE_sampling_gmm(gmm, sampN=2500, Lambda=10, nsteps=200)
x0_rev = x_traj_rev[-1]   # final positions after reverse diffusion, shape: (sampN, 2)
gmm_samples, _, _ = gmm.sample(2500)

with plt.xkcd():
  figh, axs = plt.subplots(1, 1, figsize=[6.5, 6])
  handles = []
  kdeplot(x0_rev, "Samples from Reverse Diffusion", ax=axs, handles=handles, color="blue")
  kdeplot(gmm_samples, "Samples from original GMM", ax=axs, handles=handles, color="orange")
  gmm_pdf_contour_plot(gmm, cmap="Greys", levels=20)
  plt.legend(handles=handles)
  figh.show()