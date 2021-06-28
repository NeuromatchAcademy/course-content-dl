def update_policy():
  R = 0
  rewards = []

  # Discount future rewards back to the present using gamma
  for r in policy.reward_episode[::-1]:
    R = r + policy.gamma * R
    rewards.insert(0, R)

  # Scale rewards
  rewards = torch.FloatTensor(rewards)
  rewards = (rewards - rewards.mean()) / (rewards.std() +
                                          np.finfo(np.float32).eps)

  # Calculate loss
  pg_loss = (torch.sum(torch.mul(policy.past_policy,
                              Variable(rewards)).mul(-1), -1))

  # Update network weights
  # Use zero_grad(), backward() and step() methods of the optimizer instance.
  pg_optimizer.zero_grad()
  pg_loss.backward()
  # Update the weights
  for param in policy.parameters():
      param.grad.data.clamp_(-1, 1)
  pg_optimizer.step()

  # Save and intialize episode past counters
  policy.past_loss.append(pg_loss.item())
  policy.past_reward.append(np.sum(policy.reward_episode))
  policy.past_policy = Variable(torch.Tensor())
  policy.reward_episode= []