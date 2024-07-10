import jax.numpy as jnp

class GridWorld:
    def __init__(self, height, width):
        self.width = width
        self.height = height
        self.grid = jnp.zeros((height, width))
        self.agent_pos = (0, 0)
        self.goal_pos = (height - 1, width - 1)

    def render(self):
        render_grid = self.grid.copy()
        render_grid_agent = render_grid.at[self.agent_pos].set(1)
        render_grid_agentgoal = render_grid_agent.at[self.goal_pos].set(2)
        
        print("Grid World:")
        for row in render_grid_agentgoal:
            print(" ".join(["A" if cell == 1 else "G" if cell == 2 else "." for cell in row]))
        print("\nLegend: A - Agent, G - Goal, . - Empty")

env = GridWorld(width=5, height=5)

print("Initial state:")
env.render()
