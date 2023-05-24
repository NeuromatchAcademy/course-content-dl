class GridWorldPlanner(GridWorldBase):
  """A GridWorld that finds a better policy."""

  def plan(self):
    """Define a better planner!

    This gives you a starting point by setting the proper actions in the cells
    surrounding the goal cell.

    **Assignment:** Do the rest!
    """
    super().plan()
    goal_queue = [self.goal_cell]
    goals_done = set()
    while goal_queue:
      goal = goal_queue.pop(0)  # pop from front of list
      goal_neighbours = self.get_neighbours(goal)
      goals_done.add(tuple(goal))
      for a in goal_neighbours:
        nbr = tuple(goal_neighbours[a])
        if nbr == goal:
          continue
        if nbr not in goals_done:
          if a == '<':
            self.policy[nbr[0], nbr[1]] = '>'
          elif a == '>':
            self.policy[nbr[0], nbr[1]] = '<'
          elif a == '^':
            self.policy[nbr[0], nbr[1]] = 'v'
          else:
            self.policy[nbr[0], nbr[1]] = '^'
          goal_queue.append(nbr)