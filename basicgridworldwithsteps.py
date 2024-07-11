import numpy as np
import random

class GridWorld:
    def __init__(self, height, width):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))
        self.agent_pos = (0, 0)
        self.goal_pos = (height - 1, width - 1)

    def step(self, action):
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)] # 0: up, 1: right, 2: down, 3: left. This is a list of tuples.
        new_pos = (self.agent_pos[0] + directions[action][0], # Later, 'action' will be a variable assigned a number between 0 and 3
                   self.agent_pos[1] + directions[action][1])
        if (0 <= new_pos[0] < self.height) and (0 <= new_pos[1] < self.width):
            self.agent_pos = new_pos

        reward = -1  # Small negative reward for each step
        done = False

        if self.agent_pos == self.goal_pos:
            reward = 10  # Positive reward for reaching the goal
            done = True

        return self.agent_pos, reward, done

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

# Perform some random actions
for _ in range(5): # we use '_' instead of 'i' when the index isn't used in the code
    action = random.randint(0, 3)
    next_state, reward, done = env.step(action) # This is possible only because env.step(action) returns self.agent_pos, reward, done. We're 'unpacking the return'.
    translation = {0:'Up', 1:'Right', 2:'Down', 3:'Left'}
    print(f"\nAction: {translation[action]}") # To do: Print "up" rather than "0", etc.
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    env.render()

    if done:
        print("Goal reached!")
        break