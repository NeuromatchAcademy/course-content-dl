model1 = torchvision.models.resnet18(pretrained=True)
model2 = torchvision.models.alexnet(pretrained=True)
model3 = torchvision.models.vgg16(pretrained=True)

models = {'ResNet18':model1, 'AlexNet':model2, 'VGG':model3}
learning_rates = [1e-4, 1e-4, 1e-4]

times, top_1_accuracies = [], []