# Ideas

## Biological Image Analysis

Count cells: use the [segmentation notebook](https://colab.research.google.com/github/NeuromatchAcademy/course-content-dl/blob/main/projects/Neuroscience/cellular_segmentation.ipynb) and/or other approaches.

Denoise with [noise2void](https://github.com/juglab/n2v).

Try to replace the U-net with a [W-net](https://aswali.github.io/WNet/).

Try [self-supervised cell segmentation](https://www.biorxiv.org/content/10.1101/2021.05.17.444529v1.full) with modified U-nets.

## Using image features to predict human behavior

[THINGS](https://twitter.com/martin_hebart/status/1396811812180578305): [images/behavioral data](https://osf.io/jum2f/), [paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0223792).

## Deep leakage from gradients

Background. Privacy is essential in deep learning, particularly in federated settings/environments where different clients want to train models together without sharing their private data. An adopted method of hiding clients personal information while training models collaboratively is sharing gradient updates. Gradients have been widely used in federated/collaborative learning, however it has been shown that an attacker can retrieve the exact input data simply from the shared gradients [Zhu et al](https://arxiv.org/abs/1906.08935). In this project, your task is to reimplement a gradient attack method from this paper and show that one can retrieve pixel wise correct input images that were initially used in the model training.

## Flow-based VAE

Compare [flow](https://arxiv.org/abs/1912.02762 )-based VAE to normal VAE for images? See this [post](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html).

## Self-supervised learning

Compare some [contrastive self-supervision](https://arxiv.org/abs/2004.11362
) methods on image classification problems, e.g. a vanilla supervised training vs training with self-supervision.

## Graph Neural networks

Use the GNN deepwalk embedding on some molecules dataset and use CNN to cluster the molecules. See [ref1](https://github.com/rusty1s/pytorch_geometric
) and [ref2](https://github.com/dsgiitr/graph_nets).


# Datasets

[THINGS](https://things-initiative.org/projects/things-images/)

[OpenNeuro](https://openneuro.org/public/datasets)

[Kaggle](https://www.kaggle.com/datasets)

[VisualData](https://visualdata.io/discovery)

[Models](https://models.roboflow.com/)
