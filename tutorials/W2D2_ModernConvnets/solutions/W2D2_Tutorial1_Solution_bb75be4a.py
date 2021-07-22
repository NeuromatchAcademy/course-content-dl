
"""
Challenge 1: limited data / overfitting. Limited amount of labeled data for many tasks beyond ImageNet.
Labels are expensive, for many tasks we don't have enough of them. --> Large networks will overfit.
[Solution: transfer learning: adapt networks trained on ImageNet to other tasks]

Challenge 2: hardware limitations. Making networks bigger/deeper will increase
compute and memory requirements. --> There are physical limits what can be done, and completed in reasonable time.
[Solution: more efficient architectures than standard convolutions]

Challenge 3: training deep networks "out-of-the-box" is unstable / training diverges
[Solution:.1 mechanisms like batch normalization and architectures like ResNets]

NOTE: Students cannot know any of the solutions to the problems from the material.
The intention of the question is to have them discuss and think about the
challenges (primarily the first two). The solutions are just provided for completeness for the tutors.
"""