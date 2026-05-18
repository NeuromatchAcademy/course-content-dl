"""
Discussion:

No, the denoising objective cannot be optimized to zero in most cases!

Imagine there are at least two datapoints from dataset $x_1,x_2$.
then there exist noise pattern s.t. $x_1+\sigma_t z_1=x_2+\sigma_t z_2$
In these two cases, the score network has the same input $\tilde x=x_1+\sigma_t z_1=x_2+\sigma_t z_2$,
But it needs to predict two different targets $-z_1/sigma_t$ and $-z_2/sigma_t$
Thus the optimized score will be the averaged version of $-z_1/sigma_t$, $-z_2/sigma_t$ and all other data.
""";