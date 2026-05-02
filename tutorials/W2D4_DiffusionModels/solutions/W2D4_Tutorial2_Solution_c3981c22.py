def reverse_diffusion_SDE_sampling_gmm(gmm, sampN=500, Lambda=5, nsteps=500):
  """ Using exact score function to simulate the reverse SDE to sample from distribution.

  gmm: Gausian Mixture model class defined above
  sampN: Number of samples to generate
  Lambda: the $\lambda$ used in the diffusion coefficient $g(t)=\lambda^t$
  nsteps: how many discrete steps do we use to
  """
  # initial distribution $N(0,sigma_T^2 I)$
  sigmaT2 = sigma_t_square(1, Lambda)
  xT = np.sqrt(sigmaT2) * np.random.randn(sampN, 2)
  x_traj_rev = np.zeros((*xT.shape, nsteps, ))
  x_traj_rev[:,:,0] = xT
  dt = 1 / nsteps
  for i in range(1, nsteps):
    # note the time fly back $t$
    t = 1 - i * dt

    # Sample the Gaussian noise $z ~ N(0, I)$
    eps_z = np.random.randn(*xT.shape)

    # Transport the gmm to that at time $t$ and
    gmm_t = diffuse_gmm(gmm, t, Lambda)

    # Compute the score at state $x_t$ and time $t$, $\nabla \log p_t(x_t)$
    score_xt = gmm_t.score(x_traj_rev[:, :, i-1])

    # Implement the one time step update equation
    x_traj_rev[:, :, i] = x_traj_rev[:, :, i-1] + eps_z * (Lambda ** t) * np.sqrt(dt) + score_xt * dt * Lambda**(2*t)
  return x_traj_rev


## Uncomment the code below to test your function
set_seed(42)
x_traj_rev = reverse_diffusion_SDE_sampling_gmm(gmm, sampN=2500, Lambda=10, nsteps=200)
x0_rev = x_traj_rev[:, :, -1]
gmm_samples, _, _ = gmm.sample(2500)

with plt.xkcd():
  figh, axs = plt.subplots(1, 1, figsize=[6.5, 6])
  handles = []
  kdeplot(x0_rev, "Samples from Reverse Diffusion", ax=axs, handles=handles, color="blue")
  kdeplot(gmm_samples, "Samples from original GMM", ax=axs, handles=handles, color="orange")
  gmm_pdf_contour_plot(gmm, cmap="Greys", levels=20)  # the exact pdf contour of gmm
  plt.legend(handles=handles)
  figh.show()