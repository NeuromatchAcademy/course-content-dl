
"""
Discussion:

1. Downsampling is implemented as stride in Conv2d
2. Upsampling is implemented as stride in ConvTranspose2d
3. Skip connection is by concatenation e.g.
  `self.tconv3(torch.cat([h, h3], dim=1))`
4. By adding the output of `t_mod` layers e.g.
  `h = self.tconv3(torch.cat([h, h3], dim=1)) + self.t_mod6(embed)`
5. Inspective the objective we can see the target for $s(x,t) is z/\sigma_t$.
*Pro*: If we divide the noise scale $\sigma_t$, within the network, then the neural network only need to model
the target $z$ which has the same variance ~ 1 across time $t$.

*Con*: Note this will amplify the noise and signal by a lot when t ~ 0 .
So it will have large error for low noise conditions.
The weighting function is kind of counteracting this effect,
i.e. lowering the weights of small $\sigma_t$ period in the overall loss.
""";