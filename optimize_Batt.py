from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.environ import SolverFactory, value
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df_price = pd.read_csv('Price-2019.csv')
df = pd.read_csv('Data.csv')
df_thermal = pd.read_csv('Thermal_loads.csv')

pv_generation_dict = df['PV_Generation'].to_dict()
spot_prices_dict = df_price['Price'].to_dict()
electrical_loads_dict = df['Loads'].to_dict()
thermal_loads_dict = df_thermal['Space_heating'].to_dict()

m = ConcreteModel()
m.T = RangeSet(0, 8759)

m.pv_generation = Param(m.T, initialize=pv_generation_dict)
m.spot_price = Param(m.T, initialize=spot_prices_dict)
m.electrical_load = Param(m.T, initialize=electrical_loads_dict)
m.thermal_load = Param(m.T, initialize=thermal_loads_dict)

COP_pump = 2  
stes_capacity = 5000  # In kWh
import_cost_factor = 2.5  # Cost to import is 2.5 times spot price

battery_capacity = 800 
battery_charge_efficiency = 0.99  
battery_discharge_efficiency = 0.99  

m.pv_to_grid = Var(m.T, within=NonNegativeReals)  # PV to grid
m.energy_to_store = Var(m.T, within=NonNegativeReals) # Total energy to STES
m.grid_to_elec = Var(m.T, within=NonNegativeReals) # Grid to Elec. Loads
m.grid_to_batt = Var(m.T, within=NonNegativeReals) # Grid to Elec. Loads
m.thermal_to_import = Var(m.T, within=NonNegativeReals) # District Heating to Thermal Loads
m.stes_state_of_charge = Var(m.T, within=NonNegativeReals) # SOC of STES
m.stes_to_load = Var(m.T, within=NonNegativeReals)  # STES to Thermal Loads
m.pv_to_stes = Var(m.T, within=NonNegativeReals)   # PV to STES
m.pv_to_elec = Var(m.T, within=NonNegativeReals)   # PV to Elec. Loads
m.pv_to_batt = Var(m.T, within=NonNegativeReals)   # PV to Battery
m.grid_to_stes = Var(m.T, within=NonNegativeReals)  # Grid to STES

m.stes_mode = Var(m.T, within=Binary)

m.battery_charge = Var(m.T, within=NonNegativeReals)
m.battery_discharge = Var(m.T, within=NonNegativeReals)
m.battery_state_of_charge = Var(m.T, within=NonNegativeReals)
m.battery_mode = Var(m.T, within=Binary)
  
def objective_rule(m):
    return sum(import_cost_factor * m.spot_price[t] * (m.grid_to_elec[t] + m.thermal_to_import[t] + m.grid_to_batt[t]) for t in m.T) - sum(m.spot_price[t] * m.pv_to_grid[t] for t in m.T)
m.objective = Objective(rule=objective_rule, sense=minimize)

def electrical_balance_rule_with_battery(m, t):
    return (m.pv_to_elec[t] + m.grid_to_elec[t] + m.battery_discharge[t]*(1 - m.battery_mode[t]) == m.electrical_load[t] + m.battery_charge[t]*m.battery_mode[t])
m.electrical_balance_with_battery = Constraint(m.T, rule=electrical_balance_rule_with_battery)

def battery_charging_rule(m, t):
    return m.battery_discharge[t] <= (1 - m.battery_mode[t]) * battery_capacity*0.5
m.battery_charging_constraint = Constraint(m.T, rule=battery_charging_rule)

def battery_discharging_rule(m, t):
    return m.battery_charge[t] <= m.battery_mode[t] * battery_capacity*0.5
m.battery_discharging_constraint = Constraint(m.T, rule=battery_discharging_rule)

def battery_charging_rule(m, t):
    return m.battery_charge[t] <= m.pv_to_batt[t] + m.grid_to_batt[t]
m.battery_charging_constraint = Constraint(m.T, rule=battery_charging_rule)

def battery_capacity_rule(m, t):
    return m.battery_state_of_charge[t] <= battery_capacity
m.battery_capacity_constraint = Constraint(m.T, rule=battery_capacity_rule)

def battery_min_soc_rule(m, t):
    return m.battery_state_of_charge[t] >= 0
m.battery_min_soc_constraint = Constraint(m.T, rule=battery_min_soc_rule)

def battery_state_rule(m, t):
    if t == 0:
        return m.battery_state_of_charge[t] == 0  # Initial 
    else:
        return m.battery_state_of_charge[t] == (m.battery_state_of_charge[t-1] + m.battery_charge[t]*m.battery_mode[t] - m.battery_discharge[t]*(1 - m.battery_mode[t]))
m.battery_state_constraint = Constraint(m.T, rule=battery_state_rule)

def PV_balance_rule(m, t):
    return m.pv_to_elec[t] + m.pv_to_grid[t] + m.pv_to_stes[t] + m.pv_to_batt[t] <= m.pv_generation[t]
m.PV_balance = Constraint(m.T, rule=PV_balance_rule)

def stes_energy_rule(m, t):
    return m.energy_to_store[t] <= m.pv_to_stes[t]*COP_pump 
m.stes_energy = Constraint(m.T, rule=stes_energy_rule)

def thermal_balance_rule(m, t):
    return m.stes_to_load[t] + m.thermal_to_import[t] == m.thermal_load[t]
m.thermal_balance = Constraint(m.T, rule=thermal_balance_rule)

# def grid_dh_rule(m, t):   # Check 
#     return m.grid_to_stes[t] <= m.thermal_to_import[t]
# m.grid_dh = Constraint(m.T, rule=grid_dh_rule)

def stes_state_rule(m, t):
    if t == 0:
        return m.stes_state_of_charge[t] == 0  # Initial 
    else:
        # The state of charge change is based on the mode
        return m.stes_state_of_charge[t] == (m.stes_state_of_charge[t-1] + m.energy_to_store[t] * m.stes_mode[t] - m.stes_to_load[t] * (1 - m.stes_mode[t]))
m.stes_state_constraint = Constraint(m.T, rule=stes_state_rule)

def stes_charging_rule(m, t):
    return m.stes_to_load[t] <= (1 - m.stes_mode[t]) * stes_capacity
m.stes_charging_constraint = Constraint(m.T, rule=stes_charging_rule)

def stes_discharging_rule(m, t):
    return m.energy_to_store[t] <= m.stes_mode[t] * stes_capacity
m.stes_discharging_constraint = Constraint(m.T, rule=stes_discharging_rule)

def stes_capacity_rule(m, t):
    return m.stes_state_of_charge[t] <= stes_capacity
m.stes_capacity_constraint = Constraint(m.T, rule=stes_capacity_rule)

def stes_min_soc_rule(m, t):
    return m.stes_state_of_charge[t] >= 0
m.stes_min_soc_constraint = Constraint(m.T, rule=stes_min_soc_rule)

solver = SolverFactory('gurobi')
solver.options['max_iter'] = 100
solver.options['TimeLimit'] = 200  # Seconds
results = solver.solve(m, tee=True)

pv_to_grid_values2 = [m.pv_to_grid[t].value for t in m.T]
energy_to_store_values2 = [m.energy_to_store[t].value for t in m.T]

if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
    pv_to_grid_values = [value(m.pv_to_grid[t]) for t in m.T]
    energy_to_store_values = [value(m.energy_to_store[t]) for t in m.T]    
else:
    print("An optimal solution was not found.")
    if results.solver.termination_condition == TerminationCondition.infeasible:
        print("Problem is infeasible.")
    else:
        print("Solver Status:", results.solver.status)

objective_value = value(m.objective)/1000
print(f"The cost of operation is: {objective_value}") 


pv_generation_values = [m.pv_generation[t] for t in m.T]
electrical_load_values = [m.electrical_load[t] for t in m.T]
thermal_load_values = [m.thermal_load[t] for t in m.T]

pv_to_grid_values = [m.pv_to_grid[t].value for t in m.T]
energy_to_store_values = [m.energy_to_store[t].value for t in m.T]
grid_to_elec_values = [m.grid_to_elec[t].value for t in m.T]
thermal_to_import_values = [m.thermal_to_import[t].value for t in m.T]
stes_state_of_charge_values = [m.stes_state_of_charge[t].value for t in m.T]
stes_to_load_values = [m.stes_to_load[t].value for t in m.T]


plt.figure(figsize=(15, 8))
plt.plot(pv_generation_values, label='PV Generation', color='orange')
plt.plot(electrical_load_values, label='Electrical Load', color='blue')
plt.plot(thermal_load_values, label='Thermal Load', color='green')
plt.title('PV Generation and Loads Over Time')
plt.xlabel('Hour of the Year')
plt.ylabel('Energy (kWh)')
plt.legend()
plt.show()


pv_to_grid = [value(m.pv_to_grid[index]) for index in m.T]
pv_to_elec = [value(m.pv_to_elec[index]) for index in m.T]
pv_to_stes = [value(m.pv_to_stes[index]) for index in m.T]
# grid_to_stes = [value(m.grid_to_stes[index]) for index in m.T]
grid_to_elec = [value(m.grid_to_elec[index]) for index in m.T]
thermal_to_import = [value(m.thermal_to_import[index]) for index in m.T]
stes_state_of_charge = [value(m.stes_state_of_charge[index]) for index in m.T]
stes_to_load = [value(m.stes_to_load[index]) for index in m.T]
pv_to_batt = [value(m.pv_to_batt[index]) for index in m.T]
grid_to_batt = [value(m.grid_to_batt[index]) for index in m.T]
battery_state_of_charge = [value(m.battery_state_of_charge[t]) for t in m.T]
battery_charge = [value(m.battery_charge[t]) for t in m.T]
battery_discharge = [value(m.battery_discharge[t]) for t in m.T]

thermal_load = [value(m.thermal_load[t]) for t in m.T]
electrical_load = [value(m.electrical_load[t]) for t in m.T]
pv_generation = [value(m.pv_generation[t]) for t in m.T]

thermal_balance_check = all(
    stes_to_load[t] + thermal_to_import[t] == thermal_load[t] for t in range(len(m.T))
)

pv_generation_balance_check = all(
    pv_to_elec[t] + pv_to_grid[t] + pv_to_stes[t] + pv_to_batt[t] <= pv_generation[t] for t in range(len(m.T))
)

battery_mode = [value(m.battery_mode[t]) for t in m.T]

electrical_balance_check = all(
    pv_to_elec[t] + grid_to_elec[t] + battery_discharge[t] * (1 - battery_mode[t]) == 
    electrical_load[t] + battery_charge[t] * battery_mode[t] for t in range(len(m.T))
)

print(f"Thermal balance holds for all t: {thermal_balance_check}")
print(f"PV generation balance holds for all t: {pv_generation_balance_check}")
print(f"Electrical balance holds for all t: {electrical_balance_check}")


########################################

pv_generation_difference = [pv_to_elec[t] + pv_to_grid[t] + pv_to_stes[t] + pv_to_batt[t] - pv_generation[t] for t in range(len(m.T))]
electrical_balance_difference = [(pv_to_elec[t] + grid_to_elec[t] + battery_discharge[t] * (1 - battery_mode[t])) - (electrical_load[t] + battery_charge[t] * battery_mode[t]) for t in range(len(m.T))]
thermal_balance_difference = [stes_to_load[t] + thermal_to_import[t] - thermal_load[t] for t in range(len(m.T))]

plt.figure(figsize=(10, 5))
plt.plot(pv_generation_difference, label='PV Generation Balance Difference', color='blue')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.xlabel('Hour')
plt.ylabel('Difference (kWh)')
plt.title('PV Generation Balance Difference per Hour')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(electrical_balance_difference, label='Electrical Balance Difference', color='green')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.xlabel('Hour')
plt.ylabel('Difference (kWh)')
plt.title('Electrical Balance Difference per Hour')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(thermal_balance_difference, label='Thermal Balance Difference', color='red')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.xlabel('Hour')
plt.ylabel('Difference (kWh)')
plt.title('Thermal Balance Difference per Hour')
plt.legend()
plt.show()


