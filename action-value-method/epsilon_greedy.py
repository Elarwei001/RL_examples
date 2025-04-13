import numpy as np

def e_greedy(epsilon=0.1, n_arms=10, n_steps=1000):
    # Initialize reward probabilities for each arm (unknown to the agent)
    true_rewards = np.random.normal(0, 1, n_arms)
    
    # Initialize estimates and counts
    q_values = np.zeros(n_arms)
    arm_counts = np.zeros(n_arms)
    total_reward = 0
    
    for step in range(n_steps):
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            # Explore: choose random arm
            arm = np.random.randint(n_arms)
        else:
            # Exploit: choose best arm
            arm = np.argmax(q_values)
        
        # Get reward
        reward = np.random.normal(true_rewards[arm], 1)
        total_reward += reward
        
        # Update counts and estimates
        arm_counts[arm] += 1
        q_values[arm] += (reward - q_values[arm]) / arm_counts[arm]
    
    return total_reward, q_values, arm_counts