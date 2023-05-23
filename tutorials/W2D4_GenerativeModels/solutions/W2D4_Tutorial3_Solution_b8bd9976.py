
"""
Discussion:

1. Downsampling is implemented as stride in Conv2d
2. Upsampling is implemented as stride in ConvTranspose2d
3. Skip connection is by concatenation e.g. `self.tconv3(torch.cat([h, h3], dim=1))`
4. By adding the output of `t_mod` layers `h = self.tconv3(torch.cat([h, h3], dim=1)) + self.t_mod6(embed)`
5. Inspective the objective we can see the target for s(x,t) is z/\sigma_t.
so we can divide $\sigma_t$, i.e. the noise scale within the network, thus the neural network only need to model
data of the same variance ~ 1.

Note this will amplify the noise and signal by a lot when t ~ 0 .
So it will have large error for low noise conditions.
The weighting function is kind of counteracting this effect.
""";