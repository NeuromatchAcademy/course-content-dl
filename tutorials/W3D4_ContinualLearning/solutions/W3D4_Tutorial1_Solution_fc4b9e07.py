
'''
The Permuted MNIST problem in Section 1 is an example of domain-incremental
learning.

Recall that this problem consisted of two tasks: normal MNIST (task 1) and MNIST
with permuted input images (task 2).
After learning both task, when the model is evaluated, the model is not told to
which task an image belongs (i.e., the model is not told whether the image be
classified is permuted or not), but the model also does not need to identify to
which task an image belongs (i.e., the model does not need to predict whether
the image to be classified has permuted pixels or not; it only needs to predict
the original digit displayed in the image).

Another way to motivate that this problem is an example of domain-incremental
learning, is to say that in both task 1 (normal MNIST) and task 2 (MNIST with
permuted input images), the 'type of problem' is the same (i.e., identify the
digit displayed in the original image), but the 'context' is changing (i.e.,
the order in which the image pixels are presented).
''';