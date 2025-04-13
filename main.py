from action_value_method.epsilon_greedy import e_greedy

if __name__ == "__main__":
    total_reward, final_values, counts = e_greedy()
    print(f"Total reward: {total_reward}")
    print(f"Final arm value estimates: {final_values}")
    print(f"Arm selection counts: {counts}")