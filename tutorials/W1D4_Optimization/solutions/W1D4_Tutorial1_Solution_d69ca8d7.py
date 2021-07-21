
"""
- The exact mechanism for this phenomenon is still under active research.
Existing evidence points to the following: in the overparameterized setting,
there are many more 'good configurations' (values of the model’s weights) that
lead to a low value of the objective. Furthermore, this large set of possible solutions
seems to be increasingly easy to find in the space of all possible
parameter configurations. As you increase the number of parameters, it becomes
more likely that your initialization will be close to one of these good parameter settings.

- This approach will require more memory and computation. Furthermore, we need
to always be aware of the risk of overfitting: don’t forget to do cross-validation
in order to be able to detect overfitting.
""";