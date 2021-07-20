
"""
At infinite membrane resistance, the Neuron does not leak any current out, and hence it starts firing with the slightest input current,
This shifts the transfer function towards 0, similar to ReLU activation (centered at 0).
Also, when there is minimal refractory time, the neuron can keep firing at a high input current which avoids the saturation.
""";