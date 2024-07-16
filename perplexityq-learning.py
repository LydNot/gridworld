import numpy as np
import random

class QLearning:
    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.1): # alpha, gamma, and epsilon are set to default values. epsilon is part of the action selection strategy rather than the Q-function (Q-learning update rule). parameters 'states' and 'actions' will later be assigned integer values.
        self.q_table = np.zeros((states, actions)) # states are rows, actions are columns
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.actions = actions
        self.states = states # this won't be used later on(?), but is included for completeness

    def get_action(self, state): # this function determines whether 'explore' or 'exploit' is chosen for the action
        if random.uniform(0, 1) < self.epsilon: # this will happen 10% of the time with epsilon's default value set to 0.1
            return random.randint(0, self.actions - 1)  # Explore   # return a random action from the set of actions available
        else:
            return np.argmax(self.q_table[state])  # Exploit    # self.q_table[state] will return the state with the highest q-value

# (A more sophisticated algorithm could have epsilon start high and decay over time).

    def update(self, state, action, reward, next_state): # this function updates the table's Q-values in response to a reward
        best_next_action = np.argmax(self.q_table[next_state]) # self.q_table[next_state] selects the row of the Q-table corresponding to the next state
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] # a reward is given after taking an action. self.q_table[next_state][best_next_action] retrieves the q-value of the best next action: the expected cumulative reward for taking that action. so td_target gives our new expected cumulative reward from the chosen state-action pair
        td_error = td_target - self.q_table[state][action] # td_error gives the difference between that and the previous Q-value for that state-action pair. Positive if we had an underestimate, negative if we had an overestimate.
        self.q_table[state][action] += self.alpha * td_error # this augmented assignment operator updates the Q-value. Will need to do this differently in JAX :)

# This is all we need for tabular Q-learning: a table full of zeros, a function to determine our next action, and a function to update our Q-values based on the given reward.

# Example usage
if __name__ == "__main__":  # "The if __name__ == "__main__": block in Python is a way to conditionally execute code based on whether the script is being run directly or imported as a module."
    # Initialise environment (e.g. a 4x4 grid world)
    states = 16
    actions = 4  # up, down, left, right

    # Initialise Q-learning agent
    agent = QLearning(states, actions)

    # Training loop
    episodes = 1000
    for episode in range(episodes):
        state = 0  # Start state
        done = False
        
        while not done:
            action = agent.get_action(state)
            
            # Simulate next_state and reward (replace with actual environment)
            next_state = min(state + action, states - 1)  # Simplified transition
            reward = 1 if next_state == states - 1 else 0  # Reward at goal state
            
#  If I have time, I should replace the above with gridworld with rewards :)

            # Update Q-table
            agent.update(state, action, reward, next_state)
            
            state = next_state # only possible in Python -- would need to implement differently in JAX
            done = (state == states - 1)  # Goal state

    # Print final Q-table
    print("Final Q-table:")
    print(agent.q_table)