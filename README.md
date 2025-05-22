# Active Inference Gridworld Project

This project implements an active inference framework for incentive design in stochastic gridworld environments using hiper gradient descent method.

## Overview

The system models robot agents with different potential behavioral parameters (transition probabilities and reward functions) operating in a stochastic gridworld environment. The main algorithm learns optimal side payments to maximize the conditional entropy of the posterior distribution over a agent type given a observed trajectory, while regularizing the cost of side payments. This assumes the leader has only partial observations of the agent's behavior.

## Project Structure

```
├── gridworld_example.py              # Main experiment setup and execution
├── plot_file.py                      # Convergence visualization utilities
├── generate_data_for_confusion_matrix.py  # Confusion matrix generation
├── solver/
│   └── initial_opacity_gradient_calculation.py  # Core algorithm implementation
├── mdp_env/
│   ├── gridworld_env_multi_init_states.py      # Gridworld environment classes
│   ├── hidden_markov_model_of_P2.py            
│   ├── mdp.py     
│   └── sensors.py                              # Sensors class
├── robotmdp_para/             
│   └── robotmdp_*.txt               # Robots transition parameters
└── Data/                            # Output directory for results
    ├── entropy_values_*.pkl         # Saved entropy trajectories
    ├── x_list_*                     # Saved side payment trajectories
    ├── graph_*.png                  # Convergence plots
    └── confusion_matrix/            # Confusion matrix results
```

## Key Components

### Main Files

- **`gridworld_example.py`**: Central experiment configuration and execution
  - Sets up $n \times n$ gridworld with randomly generated targets, obstacles, and sensor coverage
  - Initializes different robot types with varying transition probabilities and reward structures
  - Runs the solver
  - Generates confusion matrices

- **`plot_file.py`**: Visualization utilities for algorithm convergence
  - Plots objective function evolution
  - Tracks estimated entropy over iterations
  - Monitors side payment magnitude
  - Saves results as PNG files

- **`generate_data_for_confusion_matrix.py`**: Performance evaluation
  - Samples trajectories for each true agent type
  - Compares posterior predictions with and without optimal side payments
  - Generates confusion matrices to evaluate identification difficulty
  - Uses seaborn heatmaps for visualization

### Core Algorithm

- **`solver/initial_opacity_gradient_calculation.py`**: 
  - Implements the `InitialOpacityPolicyGradient` class
  - Gradient-based optimization of side payments
  - Batch processing and trajectory sampling

### Environment Components

- **`mdp_env/`**: Contains MDP framework classes
  - Gridworld environment in the frame of Markov Decision Process
  - Hidden Markov Model implementations
  - Sensor network modeling with configurable noise

## Algorithm Overview

The project implements the **active inference through incentive design**:

1. **Environment Setup**: Creates a stochastic gridworld with multiple agent types
2. **Sensor Network**: Deploys sensors with overlapping coverage areas (A-I + NO area)
3. **Side Payment Optimization**: Uses policy gradients to learn payments that maximize posterior entropy
4. **Performance Evaluation**: Measures identification difficulty through confusion matrices

## Configuration Parameters

### Environment
Users can either:
- **Randomly generate** environment layouts including target locations, obstacle positions, and sensor coverage areas
- **Manually configure** specific grid dimensions, target/obstacle placements, sensor network topology, and coverage parameters

*Environment parameters (`ncols`, `nrows`, `n_targets`, `n_obstacles`, `sensor_noise`, `num_sensors`) can be modified in `gridworld_example.py`*

### Agent Types
The system supports multiple robot types with configurable:
- **Transition probabilities** (α parameters) defining stochastic movement behavior - *configured in `robotmdp_para/` directory*
- **Reward structures** with different values for target states and movement penalties - *defined in `value_dict_*` variables in `gridworld_example.py`*

### Algorithm Parameters
Key parameters are configurable in `gridworld_example.py`:
- **Regularization weight** for regularization factor in objective function - *`weight` parameter in `InitialOpacityPolicyGradient`*
- **Iteration count** for gradient descent convergence - *`iter_num` parameter*
- **Batch size** for number of trajectories processed in each batch - *`batch_size` parameter*
- **Planning horizon** for number of sampled trajectories - *`V` parameter*



## Usage

### Basic Execution
```bash
python gridworld_example.py
```

This will:
1. Generate random environment layout
2. Initialize all agent types and sensor network
3. Run the solver
4. Save results to `./Data/` directory and display the convergence plot
5. Generate and display confusion matrices

### Visualization
```bash
python plot_file.py
```

Generates convergence plots based on the saved data showing:
- Objective function trajectory
- Estimated entropy evolution  
- Side payment magnitude over iterations

## Dependencies

- **Core**: `numpy`, `torch`, `random`
- **Visualization**: `matplotlib`, `seaborn`
- **Data**: `pickle`, `sklearn`
- **Custom**: `mdp_env`, `solver` modules

## Output Files

### Data Directory Structure
```
Data/
├── entropy_values_{ex_num}.pkl      # Entropy trajectory
├── x_list_{ex_num}                  # Side payment trajectory
├── theta_collection_{ex_num}        # policy parameter trajectory  
├── graph_{ex_num}.png               # Convergence plots
└── confusion_matrix/
    ├── data_no_x_trueType{i}_{ex_num}.pkl    # Baseline predictions
    ├── data_x_opt_trueType{i}_{ex_num}.pkl   # Optimized predictions
    ├── Confusion Matrix (no_x)_{ex_num}.png   # Baseline confusion matrix
    └── Confusion Matrix (x_opt)_{ex_num}.png  # Optimized confusion matrix
```

