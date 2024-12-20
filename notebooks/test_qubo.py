#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(1, '/Users/linusrandud/Documents/UoM/ERP/MscDissertation/Deep-Opt')


# In[2]:


import torch

from COProblems.MKP import MKP
from COProblems.QUBO import QUBO
from Models.DOAE import DOAE
from OptimAE import OptimAEHandler


# In[3]:


import wandb
import matplotlib.pyplot as plt


# In[ ]:





# In[4]:


# Initialize device
device = torch.device("cpu")
print(device)

# Define the file path and problem type
file_paths = ['../data/qubo/tpp_qubo20.txt']
problem_type = 'QUBO'
use_wandb = False  # Set to True if using Weights & Biases
check_constraints = False # Set to True for contraints checking inside DO


# In[5]:


jobs = {
    1: {"duration": 13, "release": 0, "deadline": 100},
    2: {"duration": 23, "release": 5, "deadline": 100},
    3: {"duration": 13, "release": 20, "deadline": 100},
    4: {"duration": 13, "release": 30, "deadline": 100},
    5: {"duration": 13, "release": 35, "deadline": 100},
    6: {"duration": 13, "release": 0, "deadline": 100},
    7: {"duration": 13, "release": 130, "deadline": 150},
    8: {"duration": 13, "release": 0, "deadline": 150},
    9: {"duration": 23, "release": 0, "deadline": 150},
    10: {"duration": 13, "release": 0, "deadline": 150},
    11: {"duration": 10, "release": 0, "deadline": 100},
    12: {"duration": 10, "release": 20, "deadline": 100},
    13: {"duration": 10, "release": 0, "deadline": 100},
    14: {"duration": 20, "release": 0, "deadline": 100},
    15: {"duration": 25, "release": 0, "deadline": 100},
    16: {"duration": 20, "release": 100, "deadline": 120},
    17: {"duration": 10, "release": 0, "deadline": 100},
    18: {"duration": 15, "release": 0, "deadline": 150},
    19: {"duration": 10, "release": 0, "deadline": 150},
    20: {"duration": 20, "release": 0, "deadline": 150},
}

# Parameters
params = {
    'change_tolerance': 20,
    'problem_size': 20,
    'pop_size': 10000,
    'dropout_prob': 0.2,
    'l1_coef': 0.0001,
    'l2_coef': 0.0001,
    'learning_rate': 0.002,
    'max_depth': 6,
    'compression_ratio': 0.8,
    'problem_instance_id': 0,
    'deepest_only': True,
    'encode': True,
    'repair_solutions': True,
    'patience': 5,  # Number of iterations to wait
    'delta_mean_population': 0.1,  # Threshold for mean population change
    'jobs': jobs,
    'check_constraints': check_constraints
}

# Initialize problem
if problem_type == 'QUBO':
    problem = QUBO(file_paths[0], params['problem_instance_id'], device)
elif problem_type == 'MKP':
    problem = MKP(file_paths[0], file_paths[1], params['problem_instance_id'], device)
else:
    raise ValueError("Unsupported problem type")

# Unpack parameters
change_tolerance = params['change_tolerance']
problem_size = params['problem_size']
pop_size = params['pop_size']
dropout_prob = params['dropout_prob']
l1_coef = params['l1_coef']
l2_coef = params['l2_coef']
lr = params['learning_rate']
max_depth = params['max_depth']
compression_ratio = params['compression_ratio']
hidden_size = problem_size

# Initialize model and handler
model = DOAE(problem_size, dropout_prob, device)
handler = OptimAEHandler(model, problem, device)

if use_wandb:
    wandb.init(project="Deep Optimization with Constraints", tags=[problem_type])
    wandb.config.update(params)
    wandb.log_artifact(file_paths[0], type='dataset')
    if problem_type == 'MKP':
        wandb.log_artifact(file_paths[1], type='dataset')

# Generate initial population
population, fitnesses = handler.generate_population(pop_size)
population, fitnesses, _, _ = handler.hilldescent(population, fitnesses, change_tolerance, params['jobs'], check_constraints)
handler.print_statistics_min(fitnesses)

total_eval = 0
depth = 0

# Store metrics for custom plotting
mean_fitnesses = []
min_max_fitnesses = []
total_evaluations = []
mean_fitness_changes = []

while True:
    if depth < max_depth:
        print("Adding layer")
        hidden_size = round(hidden_size * compression_ratio)
        model.transition(hidden_size)
        depth += 1
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    
    print("Learning from population")
    handler.learn_from_population(population, optimizer, l1_coef=l1_coef, batch_size=pop_size)
    
    print("Optimising population")
    population, fitnesses, evaluations, done = handler.optimise_solutions_min(
        population, fitnesses, change_tolerance, encode=params['encode'], repair_solutions=params['repair_solutions'], deepest_only=params['deepest_only'], jobs=params['jobs'], check_constraints=check_constraints
    )
    handler.print_statistics_min(fitnesses)

    mean_fitness = fitnesses.mean().item()
    min_max_fitness = fitnesses.min().item()
    total_eval += evaluations

    mean_fitnesses.append(mean_fitness)
    min_max_fitnesses.append(min_max_fitness)
    total_evaluations.append(total_eval)

    if use_wandb:
        wandb.log({
            "mean_fitness": mean_fitness,
            "min_max_fitness": min_max_fitness,
            "total_eval": total_eval,
            "depth": depth
        })

    print(f"Evaluations: {total_eval}")
    
    if len(mean_fitnesses) > 1:
        mean_fitness_change = abs(mean_fitnesses[-1] - mean_fitnesses[-2])
        mean_fitness_changes.append(mean_fitness_change)
        
        if len(mean_fitness_changes) >= params['patience']:
            recent_changes = mean_fitness_changes[-params['patience']:]
            if all(change < params['delta_mean_population'] for change in recent_changes):
                print(f"Stopping early due to no significant change in mean fitness over the last {params['patience']} iterations.")
                break
    
    if done:
        print(f"Optimum solution found.")
        break


# In[ ]:





# In[6]:


# Custom plot
plt.figure(figsize=(10, 6))
plt.plot(total_evaluations, mean_fitnesses, label='Mean Fitness')
plt.plot(total_evaluations, min_max_fitnesses, label='Max Fitness')
plt.axhline(y=problem.max_fitness, color='r', linestyle='--', label='Max Possible Fitness')
plt.xlabel('Evaluations')
plt.ylabel('Fitness')
plt.title('Mean and Max Fitness over Evaluations')
plt.legend()

# # Save plot to W&B
# wandb.log({"fitness_plot": wandb.Image(plt)})

# # Finish the W&B run
# wandb.finish()


# In[7]:


population


# In[8]:


fitnesses, len(fitnesses)


# In[9]:


import numpy as np


# In[10]:


def find_extreme_indices(fitnesses, mode='high'):
    """
    Returns the indices of the highest or lowest values in the list based on the mode.

    Args:
        fitnesses (list or torch.Tensor): A list or tensor of fitness values.
        mode (str): A string that can be either 'high' or 'low'. Defaults to 'high'.

    Returns:
        list: A list of indices corresponding to the extreme values.
    """
    if mode not in ['high', 'low']:
        raise ValueError("Mode should be either 'high' or 'low'")

    if mode == 'high':
        extreme_value = max(fitnesses)
    else:
        extreme_value = min(fitnesses)

    return [i for i, value in enumerate(fitnesses) if value == extreme_value]


# In[11]:


population[find_extreme_indices(fitnesses, mode='low')]


# In[12]:


def convert_tensor_to_unique_np_arrays(tensor):
    # Convert the tensor to a numpy array
    np_array = tensor.numpy()
    
    # Replace -1 with 0
    np_array[np_array == -1] = 0
    
    # Use a set to track unique arrays based on element ordering
    unique_arrays = set()
    
    # Create a list to store unique numpy arrays
    unique_np_arrays = []
    
    for arr in np_array:
        # Convert the array to a tuple (hashable type) for uniqueness check
        arr_tuple = tuple(arr)
        
        if arr_tuple not in unique_arrays:
            unique_arrays.add(arr_tuple)
            unique_np_arrays.append(arr)
    
    # Convert the list of unique arrays back to a numpy array
    unique_np_array = np.array(unique_np_arrays)
    
    return unique_np_array


# In[13]:


convert_tensor_to_unique_np_arrays(population[find_extreme_indices(fitnesses, mode='low')]), len(convert_tensor_to_unique_np_arrays(population[find_extreme_indices(fitnesses, mode='low')]))


# In[ ]:





# In[14]:


import numpy as np

# Job details as a dictionary
jobs = {
    1: {"duration": 13, "release": 0, "deadline": 100},
    2: {"duration": 23, "release": 5, "deadline": 100},
    3: {"duration": 13, "release": 20, "deadline": 100},
    4: {"duration": 13, "release": 30, "deadline": 100},
    5: {"duration": 13, "release": 35, "deadline": 100},
    6: {"duration": 13, "release": 0, "deadline": 100},
    7: {"duration": 13, "release": 130, "deadline": 150},
    8: {"duration": 13, "release": 0, "deadline": 150},
    9: {"duration": 23, "release": 0, "deadline": 150},
    10: {"duration": 13, "release": 0, "deadline": 150},
    11: {"duration": 10, "release": 0, "deadline": 100},
    12: {"duration": 10, "release": 20, "deadline": 100},
    13: {"duration": 10, "release": 0, "deadline": 100},
    14: {"duration": 20, "release": 0, "deadline": 100},
    15: {"duration": 25, "release": 0, "deadline": 100},
    16: {"duration": 20, "release": 100, "deadline": 120},
    17: {"duration": 10, "release": 0, "deadline": 100},
    18: {"duration": 15, "release": 0, "deadline": 150},
    19: {"duration": 10, "release": 0, "deadline": 150},
    20: {"duration": 20, "release": 0, "deadline": 150},
}

def check_constraints(solution, jobs):
    machine_jobs = [[], []]
    makespans = [0, 0]
    infeasible_count = 0
    
    # Distribute jobs to the respective machines
    for job_index, job_assignment in enumerate(solution):
        machine = int(job_assignment)
        job_key = job_index + 1
        job = jobs[job_key]
        machine_jobs[machine].append((job_key, job))
    
    # Validate each machine's job schedule
    for machine, assigned_jobs in enumerate(machine_jobs):
        current_time = 0
        # Sort jobs by release date first, then by deadline
        for job_key, job in sorted(assigned_jobs, key=lambda x: (x[1]['deadline'], x[1]['release'])):
            # Ensure the job starts at the earliest possible time that meets its release date
            if current_time < job['release']:
                current_time = job['release']
            current_time += job['duration']
            # Check if the job finishes before its deadline
            if current_time > job['deadline']:
                infeasible_count += 1
                print(f"Machine and Job makes it not feasible: {machine}/{job_key}")
                print(f"Current time: {current_time}, Job's deadline: {job['deadline']}")
        makespans[machine] = current_time
    
    if infeasible_count > 0:
        return False, infeasible_count, makespans
    return True, infeasible_count, makespans


# In[15]:


solutions = convert_tensor_to_unique_np_arrays(population[find_extreme_indices(fitnesses, mode='low')])
len(solutions)


# In[16]:


# Check all solutions
for i, solution in enumerate(solutions):
    feasible, infeasible_count, makespans = check_constraints(solution, jobs)
    print(f"Solution {i + 1}: Feasible = {feasible}, Infeasible Jobs = {infeasible_count}, Makespans = {makespans}")
    print('')


# In[21]:


solutions[444-1]


# In[18]:


solutions[831-1]


# In[19]:


solutions[1075-1]


# In[ ]:





# In[ ]:


# [0., 1., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 1., 0., 0., 1., 1., 0., 0.]


# In[22]:


if False:
    print('aaa')


# In[51]:


-22000 / 4


# In[7]:


22000 * 0.4


# In[ ]:




