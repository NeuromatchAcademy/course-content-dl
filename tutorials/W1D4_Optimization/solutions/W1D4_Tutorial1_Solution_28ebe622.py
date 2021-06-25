def rmsprop_update(loss, params, grad_sq, lr=1e-1, alpha=0.8):
    """Perform an RMSprop update on a collection of parameters

    Args:
        loss (tensor): A scalar tensor containing the loss whose gradient will be computed
        params (iterable): Collection of parameters with respect to which we compute gradients
        grad_sq (iterable): Moving average of squared gradients
        lr (float): Scalar specifying the learning rate or step-size for the update
        alpha (float): Moving average parameter

    """

    # Clear up gradients as Pytorch automatically accumulates gradients from
    # successive backward calls
    zero_grad(params)

    # Compute gradients on given objective
    loss.backward()

    for (par, gsq) in zip(params, grad_sq):
        # Update estimate of gradient variance
        gsq.data = alpha * gsq.data + (1-alpha) * par.grad**2

        # Update parameters
        par.data -=  lr * (par.grad / (1e-8 + gsq.data)**0.5)