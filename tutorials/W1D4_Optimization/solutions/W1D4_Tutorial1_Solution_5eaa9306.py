
"""
- The loss has very different curvatures between the weight and the bias dimensions:
a change of the same magnitude has a much bigger effect in the loss when applied to the weight than to the bias.
This leads to very slow convergence of GD in the bias dimension.

- Momentum encourages 'movement' in previously seen directions and results in
both parameters converging much faster. The oscillations you can see for the momentum
trajectories arise due to overshooting past the solution due to the momentum term.
""";