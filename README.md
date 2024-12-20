# MSc Dissertation: Deep Optimization in Operations Scheduling

## Overview
This repository contains the code and materials for my MSc Data Science dissertation project, completed at The University of Manchester. The project, titled **"Towards Deep Optimization in Operations Scheduling,"** explores the use of deep learning techniques to optimize complex scheduling problems, such as those encountered in manufacturing and logistics.

## Project Objectives
The main goals of this project are:

- To investigate the potential of deep learning models in solving operations scheduling problems.
- To integrate deep learning with optimization methods for enhanced scheduling performance.
- To evaluate the performance of the proposed approach using benchmark datasets.

## Repository Structure
The repository is organized as follows:

```
├── Deep-Opt              # Core Python package implementing the deep optimization framework
├── data                  # Contains datasets used for training and evaluation
├── notebooks             # Jupyter Notebooks for experiments and visualizations
├── .gitignore            # Specifies files to be ignored by Git
├── LICENSE               # Licensing information (MIT License)
├── README.md             # Project overview (this file)
├── requirements.txt      # Python dependencies
```

## Installation
1. Clone the repository:

   ```bash
   git clone https://github.com/linusrd/MscDissertation.git
   cd MscDissertation
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Notebooks
The `notebooks` directory contains Jupyter Notebooks that:
- Demonstrate the preprocessing steps.
- Train and evaluate deep learning models.
- Visualize results.

#### There are two main parts of the notebooks:

##### ***Making the Problem Instances***
This involves generating or preparing the scheduling problems to be solved:
- `notebooks/generate_sched_problem.ipynb`: Used to generate scheduling problem instances.
- `notebooks/check_constraints.ipynb`: Used to brute force check the feasibility of constraints.

##### ***Running the Solvers***
This involves executing the deep optimization framework and other methods to solve the scheduling problems:
- `notebooks/test_qubo.ipynb`: Uses the deep optimization framework (DO) to solve scheduling problems.
- `notebooks/test_qubo_automated.ipynb`: Automatically uses the DO framework to solve problems across multiple parameter sets (for experimentation).
- `notebooks/other_qubo_solvers.ipynb`: Uses alternative QUBO solvers from the `qubolite` package.


### Data
The `data` directory contains the datasets used in this project. Ensure that the required datasets are properly placed in this folder before running the code. Specific dataset details and preprocessing steps are described in the notebooks.

### Deep-Opt Package
The `Deep-Opt` directory contains the implementation of the deep optimization framework. This package can be used independently for other scheduling problems by modifying configuration files and parameters.

## Results
The project demonstrates that integrating deep learning models with optimization techniques can:

- Improve the efficiency of operations scheduling.
- Handle complex constraints more effectively than traditional methods.
- Provide scalable solutions for large problem instances.

Detailed results, including model performance metrics and visualizations, are available in the notebooks.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **Supervisor**: Prof Joshua Knowles and Prof Richard Allmendinger
- Special thanks to colleagues and peers who provided feedback and support throughout the project.

## Contact
For questions or feedback, please contact me at: **[linusdanardya@gmail.com](mailto:linusrd@example.com)** or through [LinkedIn](https://www.linkedin.com/in/linusdanardya).