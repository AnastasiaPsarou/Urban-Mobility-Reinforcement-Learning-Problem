import numpy as np

def custom_round(number):
    integer_part = int(number)
    decimal_part = number - integer_part
    
    if decimal_part >= 0.5:
        rounded_number = integer_part + 1
    else:
        rounded_number = integer_part
    
    return rounded_number


import numpy as np
import matplotlib.pyplot as plt

class UserEquilibrumMARLEnvironment:
    def __init__(self, Qa, Qb, ta0, tb0):
        self.Qa = Qa
        self.Qb = Qb
        self.ta0 = ta0
        self.tb0 = tb0
        self.qb = 0
        self.qa = 0
        self.qa_history = []  # List to store qa values for each episode
        self.qb_history = []  # List to store qb values for each episode
        self.rewards = []

    def calculate_travel_times(self):
        ta = self.ta0 * (1 + (self.qa / self.Qa) ** 2)  # Calculate travel time on route a
        tb = self.tb0 * (1 + (self.qb / self.Qb) ** 2)  # Calculate travel time on route b
        return ta, tb

    def step(self, action):
        if (action == 2) and (self.qb < self.Qb):
            self.qb = self.qb + 1
        else:
            self.qa = self.qa + 1

    def compute_reward(self):
        ta, tb = self.calculate_travel_times()
        reward = abs(ta - tb)

        return reward, self.qa, self.qb

    def reset(self):
        self.qa = 0
        self.qb = 0

    def record_q_values(self):
        self.qa_history.append(self.qa)
        self.qb_history.append(self.qb)

    def plot_q_values(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.qa_history, label='qa')
        plt.plot(self.qb_history, label='qb')
        plt.xlabel('Episode')
        plt.ylabel('Number of vehicles')
        plt.legend()

        # Compute histograms
        plt.figure(figsize=(8, 6))
        plt.subplot(2, 1, 1)
        plt.hist(self.qa_history, bins='auto')
        plt.xlabel('qa')
        plt.ylabel('Frequency')

        plt.subplot(2, 1, 2)
        plt.hist(self.qb_history, bins='auto')
        plt.xlabel('qb')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    def record_reward(self, reward):
        self.rewards.append(reward)

    def plot_rewards(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward vs Number of Episodes')

        # Compute histogram
        plt.figure(figsize=(8, 6))
        plt.hist(self.rewards, bins='auto')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()


class UserEquilibrumMARLAgents:
    def __init__(self, env):
        self.env = env

    def choose_action(self):
        action = np.random.choice([1, 2])

        return action

    def train(self, episodes, num_agents):
        min_reward = float("inf")
        min_qa = 0

        for _ in range(episodes):
            total_reward = 0
            self.env.reset()

            for _ in range(num_agents):
                action = self.choose_action()
                action = self.env.step(action)

            reward, qa, qb = self.env.compute_reward()

            self.env.record_q_values()  # Record qa and qb values for the episode
            self.env.record_reward(reward)  # Record rewards values for the episode

            if reward < min_reward:
                min_reward = reward
                min_qa = qa

            print(f"Reward is: {reward}, {qa} vehicles choosing route a and {qb} vehicles choosing route b")
            print("Minimum reward is:", min_reward, "and qa in this case is:", min_qa)
            print("\n\n")

        self.env.plot_q_values()  # Plot the graph after all episodes are complete
        self.env.plot_rewards()  # Plot the graph after all episodes are complete
        return min_reward, min_qa
        
###User Equilibrm Centralized


class UserEquilibriumCentralizedEnvironment:
    def __init__(self, Qa, Qb, ta0, tb0):
        self.Qa = Qa
        self.Qb = Qb
        self.ta0 = ta0
        self.tb0 = tb0
        self.Q = 1000
        self.qa = 0  # Flow on route a
        self.qb = 0  # Flow on route b

    def calculate_travel_times(self):
        ta = self.ta0 * (1 + (self.qa / self.Qa) ** 2)  # Calculate travel time on route a
        tb = self.tb0 * (1 + (self.qb / self.Qb) ** 2)  # Calculate travel time on route b
        return ta, tb

    def step(self, action):
        self.qa, self.qb = action  # Update qa and qb based on the action

        ta, tb = self.calculate_travel_times()  # Calculate travel times

        reward = reward = abs(ta - tb)  # Reward is the sum of ta * qa and tb * qb

        return reward

    def reset(self):
        self.qa = 0
        self.qb = 0


class UserEquilibriumCentralizedController:
    def __init__(self, env, epsilon=0.005):
        self.env = env
        self.epsilon = epsilon  # Exploration rate
        self.policy = np.zeros((env.Q + 1, env.Q + 1), dtype=tuple)

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            # Explore - choose a random action
            action_b = np.random.randint(1, self.env.Q)
            action_a = self.env.Q - action_b
        else:
            # Exploit - choose the action based on the current policy
            policy_state = self.policy[state[0], state[1]]
            if isinstance(policy_state, int):
                action_a = policy_state
            else:
                action_a = policy_state[0]
            action_b = self.env.Q - action_a

            # Adjust action_a and action_b to satisfy the constraint qa + qb = 1000
            action_a = max(1, action_a)
            action_b = max(1, action_b)
            diff = self.env.Q - (action_a + action_b)
            action_a += diff // 2
            action_b += diff // 2

        return action_a, action_b

    def update_policy(self, state, action):
        self.policy[state] = action

    def train(self, num_episodes):
        qa_values = []
        qb_values = []
        rewards = []
        best_reward = float('inf')
        best_qa = 0
        best_qb = 0

        for episode in range(num_episodes):
            self.env.reset()
            state = (self.env.qa, self.env.qb)
            min_reward = float('inf')

            for _ in range(self.env.Qa + 1):
                action = self.choose_action(state)
                reward = self.env.step(action)

                next_state = (self.env.qa, self.env.qb)

                self.update_policy(state, action)
                state = next_state

                if reward < min_reward:
                    min_reward = reward
                    min_qa, min_qb = self.env.qa, self.env.qb 

            qa_values.append(min_qa)
            qb_values.append(min_qb)
            rewards.append(min_reward)

            # Check if the current episode achieved the best reward
            if min_reward < best_reward:
                best_reward = min_reward
                best_qa, best_qb = min_qa, min_qb

            print(f"Episode: {episode + 1}, Total Reward: {min_reward:.2f}, qa: {min_qa}, qb: {min_qb}")

        return best_qa, best_qb, best_reward, qa_values, qb_values, rewards

        
###System Optimum Centralized

class SystemOptimumCentralizedEnvironment:
    def __init__(self, Qa, Qb, ta0, tb0):
        self.Qa = Qa
        self.Qb = Qb
        self.ta0 = ta0
        self.tb0 = tb0
        self.Q = 1000
        self.qa = 0  # Flow on route a
        self.qb = 0  # Flow on route b

    def calculate_travel_times(self):
        ta = self.ta0 * (1 + (self.qa / self.Qa) ** 2)  # Calculate travel time on route a
        tb = self.tb0 * (1 + (self.qb / self.Qb) ** 2)  # Calculate travel time on route b
        return ta, tb

    def step(self, action):
        self.qa, self.qb = action  # Update qa and qb based on the action

        ta, tb = self.calculate_travel_times()  # Calculate travel times

        reward = ta * self.qa + tb * self.qb  # Reward is the sum of ta * qa and tb * qb

        return reward

    def reset(self):
        self.qa = 0
        self.qb = 0


class SystemOptimumCentralizedAgent:
    def __init__(self, env, epsilon=0.005):
        self.env = env
        self.epsilon = epsilon  # Exploration rate
        self.policy = np.zeros((env.Q + 1, env.Q + 1), dtype=tuple)

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            # Explore - choose a random action
            action_b = np.random.randint(1, self.env.Q)
            action_a = self.env.Q - action_b
        else:
            # Exploit - choose the action based on the current policy
            policy_state = self.policy[state[0], state[1]]
            if isinstance(policy_state, int):
                action_a = policy_state
            else:
                action_a = policy_state[0]
            action_b = self.env.Q - action_a

            # Adjust action_a and action_b to satisfy the constraint qa + qb = 1000
            action_a = max(1, action_a)
            action_b = max(1, action_b)
            diff = self.env.Q - (action_a + action_b)
            action_a += diff // 2
            action_b += diff // 2

        return action_a, action_b

    def update_policy(self, state, action):
        self.policy[state] = action

    def train(self, num_episodes):
        qa_values = []
        qb_values = []
        rewards = []
        best_reward = float('inf')
        best_qa = 0
        best_qb = 0

        for episode in range(num_episodes):
            self.env.reset()
            state = (self.env.qa, self.env.qb)
            min_reward = float('inf')

            for _ in range(self.env.Qa + 1):
                action = self.choose_action(state)
                reward = self.env.step(action)

                next_state = (self.env.qa, self.env.qb)

                self.update_policy(state, action)
                state = next_state

                if reward < min_reward:
                    min_reward = reward
                    min_qa, min_qb = self.env.qa, self.env.qb #int(action[0]), int(action[1])

            qa_values.append(min_qa)
            qb_values.append(min_qb)
            rewards.append(min_reward)

            # Check if the current episode achieved the best reward
            if min_reward < best_reward:
                best_reward = min_reward
                best_qa, best_qb = min_qa, min_qb

            print(f"Episode: {episode + 1}, Total Reward: {min_reward:.2f}, qa: {min_qa}, qb: {min_qb}")

        return best_qa, best_qb, best_reward, qa_values, qb_values, rewards
      
        
###System Optimum MARL
import matplotlib.pyplot as plt

class SystemOptimumMARLEnvironment:
    def __init__(self, Qa, Qb, ta0, tb0):
        self.Qa = Qa
        self.Qb = Qb
        self.ta0 = ta0
        self.tb0 = tb0
        self.qb = 0
        self.qa = 0
        self.qa_history = []  # List to store qa values for each episode
        self.qb_history = []  # List to store qb values for each episode
        self.rewards = []

    def calculate_travel_times(self):
        ta = self.ta0 * (1 + (self.qa / self.Qa) ** 2)  # Calculate travel time on route a
        tb = self.tb0 * (1 + (self.qb / self.Qb) ** 2)  # Calculate travel time on route b
        return ta, tb

    def step(self, action):
        if (action == 2) and (self.qb < self.Qb):
            self.qb = self.qb + 1
        else:
            self.qa = self.qa + 1

    def compute_reward(self):
        ta, tb = self.calculate_travel_times()
        reward = ta * self.qa + tb * self.qb

        return reward, self.qa, self.qb

    def reset(self):
        self.qa = 0
        self.qb = 0

    def record_q_values(self):
        self.qa_history.append(self.qa)
        self.qb_history.append(self.qb)

    def plot_q_values(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.qa_history, label='qa')
        plt.plot(self.qb_history, label='qb')
        plt.xlabel('Episode')
        plt.ylabel('Number of vehicles')
        plt.legend()

        # Compute histograms
        plt.figure(figsize=(8, 6))
        plt.subplot(2, 1, 1)
        plt.hist(self.qa_history, bins='auto')
        plt.xlabel('qa')
        plt.ylabel('Frequency')

        plt.subplot(2, 1, 2)
        plt.hist(self.qb_history, bins='auto')
        plt.xlabel('qb')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    def record_reward(self, reward):
        self.rewards.append(reward)

    def plot_rewards(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward vs Number of Episodes')

        # Compute histogram
        plt.figure(figsize=(8, 6))
        plt.hist(self.rewards, bins='auto')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()
        
class SystemOptimumMARLAgents:
    def __init__(self, env):
        self.env = env

    def choose_action(self):
        action = np.random.choice([1, 2])

        return action

    def train(self, episodes, num_agents):
        self.env.reset()
        min_reward = float("inf")
        min_qa = 0

        for i in range(episodes):
            total_reward = 0
            self.env.reset()

            for _ in range(num_agents):
                action = self.choose_action()
                action = self.env.step(action)

            reward, qa, qb = self.env.compute_reward()

            if min_reward > reward:
                min_reward = reward
                min_qa = qa

            self.env.record_q_values()  # Record qa and qb values for the episode
            self.env.record_reward(reward)
            print(f"Reward is: {reward}, {qa} vehicles choosing route a and {qb} vehicles choosing route b")
            print("Minimum reward is:", min_reward, "and qa in this case is: ", min_qa)
            print("\n\n")

        self.env.plot_q_values()  # Plot the graph after all episodes are complete
        self.env.plot_rewards()
        return min_reward, min_qa
