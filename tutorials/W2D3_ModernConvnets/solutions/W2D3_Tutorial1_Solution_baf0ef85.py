def create_models(weights):
  """
  Creates models

  Args:
    weights: list of strings
      If True, load pretrained models.

  Returns:
    models: dict
      Log of models
    lr_rates: list
      Log of learning rates
  """
  # Load three pretrained models from torchvision.models
  # [these are just examples, other models are possible as well]
  model1 = torchvision.models.resnet18(weights=weights[0])
  model2 = torchvision.models.alexnet(weights=weights[1])
  model3 = torchvision.models.vgg19(weights=weights[2])

  models = {'ResNet18': model1, 'AlexNet': model2, 'VGG19': model3}
  lr_rates = [1e-4, 1e-4, 1e-4]

  return models, lr_rates


weight_list = ['ResNet18_Weights.DEFAULT', 'AlexNet_Weights.DEFAULT', 'VGG19_Weights.DEFAULT']
## Uncomment below to test your function
models, lr_rates = create_models(weights=weight_list)
times, top_1_accuracies = run_models(models, lr_rates)
with plt.xkcd():
  plot_acc_speed(times, top_1_accuracies, models)