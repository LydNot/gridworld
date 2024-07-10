import numpy as np

class GridWorld:
    def __init__(self, height, width):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))
        self.agent_pos = (0, 0)
        self.goal_pos = (height - 1, width - 1)

    def render(self):
        render_grid = self.grid.copy()
        render_grid[self.agent_pos] = 1
        render_grid[self.goal_pos] = 2
        
        print("Grid World:")
        for row in render_grid:
            print(" ".join(["A" if cell == 1 else "G" if cell == 2 else "." for cell in row]))
        print("\nLegend: A - Agent, G - Goal, . - Empty")

env = GridWorld(width=5, height=5)

print("Initial state:")
env.render()
