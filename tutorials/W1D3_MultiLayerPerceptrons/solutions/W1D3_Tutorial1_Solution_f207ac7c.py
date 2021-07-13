
"""
At infinite membrane resistance, the Neuron does not leak any current out, and hence it starts firing with the slightest input current,
This shifts the transfer function towards 0, similar to ReLU activation (centered at 0).
In addition, dV/dt = (-V/R + I)/C simplifis to dV/dt = I/C which results in the firing rate to be only a linear function of input current.
In summary, infinite membrane potential makes a neuron's transfer function a ReLU function.
""";