def create_models(pretrained=False):
  """
  Creates models

  Args:
    pretrained: boolean
      If True, load pretrained models

  Returns:
    models: dict
      Log of models
    lr_rates: list
      Log of learning rates
  """
  # Load three pretrained models from torchvision.models
  # [these are just examples, other models are possible as well]
  model1 = torchvision.models.resnet18(pretrained=pretrained)
  model2 = torchvision.models.alexnet(pretrained=pretrained)
  model3 = torchvision.models.vgg19(pretrained=pretrained)

  models = {'ResNet18': model1, 'AlexNet': model2, 'VGG-19': model3}
  lr_rates = [1e-4, 1e-4, 1e-4]

  return models, lr_rates


## Uncomment below to test your function
models, lr_rates = create_models(pretrained=True)
times, top_1_accuracies = run_models(models, lr_rates)
plot_acc_speed(times, top_1_accuracies, models)