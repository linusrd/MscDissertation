import argparse
import json
import torch
import matplotlib.pyplot as plt
import wandb
import sys
print(sys.prefix)
sys.path.insert(1, '/Users/linusrandud/Documents/UoM/ERP/MscDissertation/Deep-Opt')
from COProblems.MKP import MKP
from COProblems.QUBO import QUBO
from Models.DOAE import DOAE
from OptimAE import OptimAEHandler

def parse_args():
    parser = argparse.ArgumentParser(description="Deep Optimization with Non-Linear Transitions")
    parser.add_argument('--params', type=str, required=True, help='Path to the parameters JSON file')
    parser.add_argument('--file_paths', type=str, nargs='+', required=True, help='Paths to the problem instance files')
    parser.add_argument('--problem_type', type=str, choices=['QUBO', 'MKP'], required=True, help='Type of the problem instance')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the computations on (cpu or cuda)')
    return parser.parse_args()

def read_params(file_path):
    with open(file_path, 'r') as file:
        params = json.load(file)
    return params

def main():
    args = parse_args()
    params = read_params(args.params)
    device = torch.device(args.device)

    if args.problem_type == 'QUBO':
        problem = QUBO(args.file_paths[0], params['problem_instance_id'], device)
    elif args.problem_type == 'MKP':
        problem = MKP(args.file_paths[0], args.file_paths[1], params['problem_instance_id'], device)
    else:
        raise ValueError("Unsupported problem type")

    if args.use_wandb:
        wandb.init(project="Deep Optimization", tags=[args.problem_type])
        wandb.log_artifact(args.file_paths[0], type='dataset')
        if args.problem_type == 'MKP':
            wandb.log_artifact(args.file_paths[1], type='dataset')

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

    if args.use_wandb:
        wandb.config.update({
            "change_tolerance": change_tolerance,
            "problem_size": problem_size,
            "pop_size": pop_size,
            "dropout_prob": dropout_prob,
            "l1_coef": l1_coef,
            "l2_coef": l2_coef,
            "learning_rate": lr,
            "compression_ratio": compression_ratio,
            "max_depth": max_depth,
        })
        wandb.watch(model, log='all')

    # Generate initial population
    population, fitnesses = handler.generate_population(pop_size)
    population, fitnesses, _, _ = handler.hillclimb(population, fitnesses, change_tolerance)
    handler.print_statistics(fitnesses)

    total_eval = 0
    depth = 0

    # Store metrics for custom plotting
    mean_fitnesses = []
    max_fitnesses = []
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
        population, fitnesses, evaluations, done = handler.optimise_solutions(
            population, fitnesses, change_tolerance, encode=params['encode'], repair_solutions=params['repair_solutions'], deepest_only=params['deepest_only']
        )
        handler.print_statistics(fitnesses)

        mean_fitness = fitnesses.mean().item()
        max_fitness = fitnesses.max().item()
        total_eval += evaluations

        mean_fitnesses.append(mean_fitness)
        max_fitnesses.append(max_fitness)
        total_evaluations.append(total_eval)

        if args.use_wandb:
            wandb.log({
                "mean_fitness": mean_fitness,
                "max_fitness": max_fitness,
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

    # Custom plot
    plt.figure(figsize=(10, 6))
    plt.plot(total_evaluations, mean_fitnesses, label='Mean Fitness')
    plt.plot(total_evaluations, max_fitnesses, label='Max Fitness')
    plt.axhline(y=problem.max_fitness, color='r', linestyle='--', label='Max Possible Fitness')
    plt.xlabel('Evaluations')
    plt.ylabel('Fitness')
    plt.title('Mean and Max Fitness over Evaluations')
    plt.legend()

    # Save plot to W&B
    wandb.log({"fitness_plot": wandb.Image(plt)})

    # Finish the W&B run
    wandb.finish()

if __name__ == "__main__":
    main()
