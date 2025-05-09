
"""
Stochastic Gradient Descent (SGD): Performs updates one example at a time.
Momentum: Helps accelerate SGD in the relevant direction and dampens
oscillations specially ravines.
RMSProp: Allows each parameter to be updated at an 'appropriate' rate decided
based on magnitudes of past recent updates;
i.e., areas where the surface curves much more steeply in one dimension than
in another, which are common around local optima.

Robustness: RMSProp > Momentum > SGD
Since, each example affects SGD by updating hyperparameters, it's not
considered very robust.
Adagrad greatly improved the robustness of SGD and is used for training
large-scale neural nets.
Momentum is quite robust: he momentum term increases for dimensions whose
gradients point in the same directions
and reduces updates for dimensions whose gradients change directions.
RMSProp is very robust; This combines the idea of only using the sign of
the gradient with the idea of adapting the step size separately
for each weight in a mini-batch.

Generally, non-adaptive methods consistently produce more robust models
than adaptive methods. Refer https://arxiv.org/pdf/1911.03784.pdf - for more details
""";