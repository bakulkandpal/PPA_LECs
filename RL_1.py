# With simple (linear) function approximation and TD learning
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

alpha = 0.1  
gamma = 1  # Discount factor (keep > 0.5, and <1)
n_episodes = 20  # Number of episodes, represent years without actual data.
initial_epsilon = 0.3  # Starting value for epsilon
min_epsilon = 0.01    # Minimum value epsilon
epsilon_decay = (initial_epsilon - min_epsilon) / n_episodes

epsilon = initial_epsilon

n_hours = 8760
n_actions = 2  # 0 = Store energy to STES, 1 = Sell energy outside LEC
charge_efficiency = 0.98  
discharge_efficiency = 0.98  
storage_capacity = 4000  # in kWh (check in thermal units)
overflow_penalty = -150  # Penalty for STES
underflow_penalty = -150  

df_price = pd.read_csv('Price_2019.csv')
df = pd.read_csv('Data.csv')
df_thermal = pd.read_csv('Thermal_loads.csv')

pv_generation = df['PV_Generation'].values
spot_prices = df_price['Price'].values
loads=df['Loads'].values
thermal_loads=df_thermal['Space_heating']

pv_generation=pv_generation*1  # For case studies. Higher PV would result in higher storage.

w = np.zeros((n_hours, n_actions))

def phi(hour, action):
    feature_vector = np.zeros((n_hours, n_actions))
    feature_vector[hour, action] = 1   # One-hot encoding for the state-action pair
    return feature_vector.flatten()

def q_hat(hour, action, w):
    return np.dot(phi(hour, action), w.flatten())  # Approximation, linear function with weights


all_storage_levels = []
all_policies = []
all_rewards=[]
policy_weight_save=[]

storage_levels = np.zeros(n_hours)
storage_levels[0] = storage_capacity*0.5  # Initial storage state in kWh, first year, first hour.
w_storage = np.zeros((n_episodes, n_hours, n_actions))
    
for episode in range(n_episodes):
    rewards=[]
    episode_policy = np.zeros(n_hours)
    q_values_store = np.zeros((n_hours, n_actions))
    if episode > 0:
        storage_levels[0] = 0
    for hour in range(n_hours):
        storage_before_action = storage_levels[hour]      
        if np.random.rand() < epsilon:
            action = np.random.choice(n_actions)  # Exploration
        else:
            q_values = [q_hat(hour, a, w) for a in range(n_actions)]
            #action = np.argmax(q_values)
            action = np.argmax(q_values) if q_values[0] != q_values[1] else 1
        cumulative_elec=max(pv_generation[hour]-loads[hour],0)
        cumulative_therm=storage_before_action-thermal_loads[hour]
        if action == 1:  # i.e. selling of PV
            energy_sold = cumulative_elec
            reward = spot_prices[hour] * energy_sold * 0.01
        else:  # i.e. store energy in STES
            energy_stored = min(cumulative_elec, storage_capacity - storage_before_action)
            storage_levels[hour] += energy_stored * charge_efficiency *2  # Update storage after storing. Multiplied by 2 considering COP of heat pumps
            reward = 0  

        if storage_levels[hour] > storage_capacity:
            reward = overflow_penalty
            storage_levels[hour] = storage_capacity
        elif storage_levels[hour] < 0:
            reward = underflow_penalty
            storage_levels[hour] = 0  
        
        if storage_levels[hour] >= thermal_loads[hour]:
            reward+=0
            storage_levels[hour] -= thermal_loads[hour]
        elif storage_levels[hour] < thermal_loads[hour]:
            deficit_thermal=thermal_loads[hour]-storage_levels[hour]
            reward-=spot_prices[hour]*deficit_thermal*0.01*3  # Assuming taxes etc. make import-cost 3 times.
            storage_levels[hour] = 0
            
        if hour < n_hours - 1:
            storage_levels[hour + 1] = storage_levels[hour]

        next_hour = (hour + 1) % n_hours
        q_values_next_hour = [q_hat(next_hour, a, w) for a in range(n_actions)]
        best_next_action = np.argmax(q_values_next_hour)

        td_error = reward + gamma * q_hat(next_hour, best_next_action, w) - q_hat(hour, action, w)

        w[hour, action] += alpha * td_error
        episode_policy[hour] = action
        q_values_store[hour, :] = q_values
        rewards.append(reward)
        
    all_storage_levels.append(storage_levels.copy())
    all_policies.append(episode_policy.copy())
    all_rewards.append(rewards.copy())
    policy_weight=np.argmax(w, axis=1)
    policy_weight_save.append(policy_weight.copy())
    w_storage[episode] = w
    epsilon = max(min_epsilon, epsilon - epsilon_decay)

policy = np.argmax(w, axis=1)