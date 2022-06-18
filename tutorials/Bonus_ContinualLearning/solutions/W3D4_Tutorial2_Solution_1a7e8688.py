class Multi_task_model(nn.Module):
  """
  Defines Multi-task model
  """

  def __init__(self, pretrained=True, num_tasks=4, load_file=None,
               num_labels_per_task=[2, 2, 2, 5]):
    """
    Initialize parameters of the multi-task model

    Args:
      pretrained: boolean
        If true, load pretrained model
      num_tasks: int
        Number of tasks
      load_file: string
        If specified, load requisite file [default: None]
      num_labels_per_task: list
        Specifies number of labels per task

    Returns:
      Nothing
    """
    super(Multi_task_model, self).__init__()
    self.backbone = models.resnet18(pretrained=pretrained)  # You can play around with different pre-trained models
    if load_file:
      self.backbone.load_state_dict(torch.load(load_file))
    self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1]))  # Remove the last fully connected layer

    if pretrained:
      for param in self.backbone.parameters():
        param.requires_grad = False

    self.fcs = []

    self.num_tasks = num_tasks

    for i in range(self.num_tasks):
      self.fcs.append(nn.Sequential(
          nn.Linear(512, 128),
          nn.ReLU(),
          nn.Dropout(0.4),
          nn.Linear(128, num_labels_per_task[i]),
          ################################
          # Add more layers if you want! #
          ################################
          nn.Softmax(dim=1),
      ))

      self.fcs = nn.ModuleList(self.fcs)

  def forward(self, x):
    """
    Forward pass of multi-task model

    Args:
      x: torch.tensor
        Input Data

    Returns:
      outs: list
        Fully connected layer outputs for each task
    """
    x = self.backbone(x)
    x = torch.flatten(x, 1)
    outs = []
    for i in range(self.num_tasks):
      outs.append(self.fcs[i](x))
    return outs


# Add event to airtable
atform.add_event('Coding Exercise 2.2: Creating a Multi-Task model')