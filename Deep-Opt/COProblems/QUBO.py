import torch

from COProblems import QUBO_populate_function
from COProblems.OptimisationProblem import OptimisationProblem


class QUBO(OptimisationProblem):
    """
    A class that implements the Quadratic Unconstrained Binary Optimisation problem.
    """
    def __init__(self, file: str, id: int, device: torch.device):
        """
        Constructor method for QUBO. Loads in the Q matrix for a particular instance of QUBO.

        Args:
            file: str
                The file to extract a QUBO instance from.
            id: int
                The problem instance ID.
            device: torch.device:
                The device that the problem is run on.
        """
        self.Q = QUBO_populate_function.QUBOpopulate(file, id)
        self.Q = self.Q.to(dtype=torch.float32, device=device)
        super().__init__(device)
        self.jobs = {}

    def fitness(self, x: torch.Tensor, check_constraints: bool, penalty_mult: int) -> torch.Tensor:
        """
        Calculate the fitness of any assignment of items.

        Args:
            x: torch.Tensor
                The solutions to have their fitness calculated, where each element is a '1' to 
                represent a 1 and a '-1' to represent a 0.
        
        Returns:
            The fitnesses of each solution.
        """
        x = (x + 1) / 2
        mul1 = x.matmul(self.Q)
        mul2 = (mul1 * x).sum(dim=1)
        
        if check_constraints:
            # Calculate penalties
            penalties = self.calculate_penalties(x)
            
            # Incorporate penalties into fitness
            fitness = mul2 + (penalty_mult*penalties) #arbitrary constant
            return fitness
        else:
            return mul2
    
    def calculate_penalties(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the penalties for constraint violations.

        Args:
            x: torch.Tensor
                The solutions to check for constraint violations.
        
        Returns:
            penalties: torch.Tensor
                Penalties for each solution. Shape N.
        """
        penalties = torch.zeros(x.size(0), device=self.device)
        for i, solution in enumerate(x):
            machine_jobs = [[], []]
            makespans = [0, 0]
            
            for job_index, job_assignment in enumerate(solution):
                machine = int(job_assignment)
                job_key = job_index + 1
                job = self.jobs[job_key]
                machine_jobs[machine].append((job_key, job))
            
            for machine, assigned_jobs in enumerate(machine_jobs):
                current_time = 0
                for job_key, job in sorted(assigned_jobs, key=lambda x: (x[1]['deadline'], x[1]['release'])):
                    if current_time < job['release']:
                        current_time = job['release']
                    current_time += job['duration']
                    if current_time > job['deadline']:
                        penalties[i] += 1  # Increment penalty for each violation
        return penalties
    
    def is_valid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Determines whether the given solutions violates any constraints on the problem.

        Args:
            x: torch.Tensor
                The solutions which will be tested.

        Returns:
            A tensor where each value is True due to QUBO being unconstrained.
        """
        return torch.full((x.shape[0],), True)
    
    def random_solution(self, pop_size: int) -> torch.Tensor:
        """
        Generates a population of random problem solutions.

        Args:
            pop_size: int
                The size of the population of solutions.

        Returns:
            The random solutions.
        """
        return torch.randint(0,2,(pop_size,self.Q.shape[0]), device=self.device, dtype=torch.float32) * 2 -1
    
    def repair(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repairs solutions so that they remain valid.

        Args:
            x: torch.Tensor
                The solutions to be repaired.
        
        Returns:
            The repaired solutions.
        """
        return x