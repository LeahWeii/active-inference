from mdp_env.gridworld_env_multi_init_states import *
from generate_data_for_confusion_matrix import *
import random

ex_num = 0

seed = 0
random.seed(seed)
print(f'seed={seed}')

# Initial set-up for 5 types of agent in a stochastic gridworld
ncols = 10
nrows = 10
n_targets = 12
n_obstacles = 20
n_modify=20


def generate_random_indices(ncols, nrows, n_targets, n_obstacles, n_modify):
    total_states = ncols * nrows
    all_indices = list(range(total_states))

    # Randomly select target indices
    targets = random.sample(all_indices, n_targets)

    # Split targets into two non-empty sets
    split_point = random.randint(1, n_targets - 1)
    random.shuffle(targets)
    targets_set_high_reward = targets[:split_point]
    targets_set_low_reward = targets[split_point:]

    remaining_indices = list(set(all_indices) - set(targets))

    # Randomly select obstacle indices (cannot overlap with targets)
    obstacles = random.sample(remaining_indices, n_obstacles)

    remaining_indices = list(set(remaining_indices) - set(obstacles))

    # Randomly select initial state (not in targets or obstacles)
    initial_state = random.choice(remaining_indices)
    initial_state = {initial_state}

    # For modify_list, sample state indices from non-obstacle states
    valid_states_for_sampling = list(set(all_indices) - set(obstacles))
    n_actions = 4

    # Make sure we don't try to sample more states than available
    n_modify = min(n_modify, len(valid_states_for_sampling))

    sampled_states = random.sample(valid_states_for_sampling, n_modify)
    modify_list = []
    for s in sampled_states:
        modify_list.extend([s * n_actions + a for a in range(n_actions)])

    return targets, targets_set_low_reward, targets_set_high_reward, obstacles, initial_state, modify_list


#randomly generate the locations, or you can specify your envrionment
targets, targets_low_reward, targets_high_reward, obstacles, initial, modify_list = generate_random_indices(ncols, nrows, n_targets, n_obstacles, n_modify)

print(f'targets={targets}')
print(f'targets_low_reward={targets_low_reward}')
print(f'targets_high_reward={targets_high_reward}')
print(f'obstacles={obstacles}')
print(f'possible initial states={initial}')
print(f'modify_list={modify_list}')

unsafe_u = []

#initialize uniform initial distribution
initial_dist = dict([])
for state in range(ncols*nrows):
    if state in initial:
        initial_dist[state] = 1 / len(initial)
    else:
        initial_dist[state] = 0

def generate_sensor_areas(grid_size, coverage_percent=90, num_sensors=10):
    """
    Generate random sensor areas (e.g., A to J) that cover approximately
    the specified percentage of a gridworld.

    Parameters:
    grid_size: Tuple (width, height) defining the grid dimensions
    coverage_percent: Target percentage of grid to cover (default 90%)
    num_sensors: Number of distinct sensors (default 10)

    Returns:
    Tuple: sensor_sets (list of sets for each sensor) + NO area set
    """
    width, height = grid_size
    total_cells = width * height

    # Create a set of all grid cells
    all_cells = set((x, y) for x in range(width) for y in range(height))

    # Calculate number of cells to cover
    cells_to_cover = int(total_cells * coverage_percent / 100)

    # Randomly select cells to cover
    cells_to_distribute = random.sample(list(all_cells), cells_to_cover)

    # Generate sensor labels (e.g., 'A', 'B', ..., up to num_sensors)
    sensor_labels = [chr(ord('A') + i) for i in range(num_sensors)]
    sets = {label: set() for label in sensor_labels}

    # Distribute cells to sensors
    for i, cell in enumerate(cells_to_distribute):
        sensor_label = sensor_labels[i % num_sensors]
        sets[sensor_label].add(cell)

    # Convert to flat indices
    sensor_sets_flat = [
        {y * width + x for x, y in sets[label]} for label in sensor_labels
    ]

    # Calculate NO area
    all_sensor_cells = set().union(*sensor_sets_flat)
    setNO = set(range(total_cells)) - all_sensor_cells

    # Return all sensor sets + the NO set
    return tuple(sensor_sets_flat) + (setNO,)



#specify sensor noise
sensor_noise = 0.05
print(f'sensor_noise = {sensor_noise}')

#specify number of sensors
num_sensors = 9  # or any other number of sensors you choose
sensor_labels = [chr(ord('A') + i) for i in range(num_sensors)]

# Generate sensor areas
sensor_sets = generate_sensor_areas((ncols, nrows), coverage_percent=95, num_sensors=num_sensors)
*sensor_coverage_list, setNO = sensor_sets

# Print each sensor's coverage
for label, coverage in zip(sensor_labels, sensor_coverage_list):
    print(f"set{label} = {coverage}")

# Print NO set
print(f"setNO = {setNO}")

# Initialize sensor coverage dictionary
sensor_coverage_dict = {label: sensor_coverage_list[i] for i, label in enumerate(sensor_labels)}
sensor_coverage_dict["NO"] = setNO

# Sensor setup
sensor_net = Sensor()
sensor_net.sensors = set(sensor_coverage_dict.keys())

# Set coverage for all sensors
for label, coverage in sensor_coverage_dict.items():
    sensor_net.set_coverage(label, coverage)


#initialize differenrt types of agents with different transition and reward, but same initial state and environment
robot_ts_1 = read_from_file_MDP_old('./robotmdp_para/robotmdp_1.txt')#robot 1 alpha = 0.9
robot_ts_2 = read_from_file_MDP_old('./robotmdp_para/robotmdp_2.txt')#robot 2 alpha = 0.6
robot_ts_3 = read_from_file_MDP_old('./robotmdp_para/robotmdp_3.txt')#robot 2 alpha = 0.9
robot_ts_4 = read_from_file_MDP_old('./robotmdp_para/robotmdp_4.txt')#robot 2 alpha = 0.7
robot_ts_5 = read_from_file_MDP_old('./robotmdp_para/robotmdp_5.txt')#robot 2 alpha = 1

agent_gw_1 = GridworldGui(initial, nrows, ncols, robot_ts_1, targets, obstacles, unsafe_u, initial_dist)
agent_gw_1.mdp.get_supp()
agent_gw_1.mdp.gettrans()
agent_gw_1.mdp.get_reward()

agent_gw_2 = GridworldGui(initial, nrows, ncols, robot_ts_2, targets, obstacles, unsafe_u, initial_dist)
agent_gw_2.mdp.get_supp()
agent_gw_2.mdp.gettrans()
agent_gw_2.mdp.get_reward()

agent_gw_3 = GridworldGui(initial, nrows, ncols, robot_ts_3, targets, obstacles, unsafe_u, initial_dist)
agent_gw_3.mdp.get_supp()
agent_gw_3.mdp.gettrans()
agent_gw_3.mdp.get_reward()

agent_gw_4 = GridworldGui(initial, nrows, ncols, robot_ts_4, targets, obstacles, unsafe_u, initial_dist)
agent_gw_4.mdp.get_supp()
agent_gw_4.mdp.gettrans()
agent_gw_4.mdp.get_reward()

agent_gw_5 = GridworldGui(initial, nrows, ncols, robot_ts_5, targets, obstacles, unsafe_u, initial_dist)
agent_gw_5.mdp.get_supp()
agent_gw_5.mdp.gettrans()
agent_gw_5.mdp.get_reward()

#specify the reward of each type of agent
value_dict_1 = dict()
for state in agent_gw_1.mdp.states:
    if state in targets_low_reward:
        value_dict_1[state] = 1
    elif state in targets_high_reward:
        value_dict_1[state] = 2
    else:
        value_dict_1[state] = -0.5

value_dict_2 = dict()
for state in agent_gw_2.mdp.states:
    if state in targets_low_reward:
        value_dict_2[state] = 1
    elif state in targets_high_reward:
        value_dict_2[state] = 2
    else:
        value_dict_2[state] = -2

value_dict_3 = dict()
for state in agent_gw_3.mdp.states:
    if state in targets_low_reward:
        value_dict_3[state] = 0.1
    elif state in targets_high_reward:
        value_dict_3[state] = 4
    else:
        value_dict_3[state] = 0

value_dict_4 = dict()
for state in agent_gw_4.mdp.states:
    if state in targets_low_reward:
        value_dict_4[state] = 1
    elif state in targets_high_reward:
        value_dict_4[state] = 1
    else:
        value_dict_4[state] = -1

value_dict_5 = dict()
for state in agent_gw_5.mdp.states:
    if state in targets_low_reward:
        value_dict_5[state] = 1
    elif state in targets_high_reward:
        value_dict_5[state] = 1
    else:
        value_dict_5[state] = -0.1


#initialize sidepayment
side_payment = {}
for state in agent_gw_1.mdp.states:
    s_idx = agent_gw_1.mdp.states.index(state)
    side_payment[state] = {}
    for action in agent_gw_1.mdp.actlist:
        a_idx = agent_gw_1.mdp.actlist.index(action)
        side_payment[s_idx][a_idx] = 0


#create hmm for each type of agent
hmm_1 = HiddenMarkovModelP2(agent_gw_1.mdp, sensor_net, side_payment, modify_list, value_dict=value_dict_1)
hmm_2 = HiddenMarkovModelP2(agent_gw_2.mdp, sensor_net, side_payment, modify_list, value_dict=value_dict_2)
hmm_3 = HiddenMarkovModelP2(agent_gw_3.mdp, sensor_net, side_payment, modify_list, value_dict=value_dict_3)
hmm_4 = HiddenMarkovModelP2(agent_gw_4.mdp, sensor_net, side_payment, modify_list, value_dict=value_dict_4)
hmm_5 = HiddenMarkovModelP2(agent_gw_5.mdp, sensor_net, side_payment, modify_list, value_dict=value_dict_5)
hmm_list = [hmm_1, hmm_2, hmm_3, hmm_4, hmm_5]

#run the algorithm
masking_policy_gradient = InitialOpacityPolicyGradient(hmm_list=hmm_list, ex_num=ex_num, weight=1e-01, sp = 0, iter_num=200, batch_size=10, V=200,
                                                       T=ncols+nrows)
masking_policy_gradient.solver()

#plot confusion matrix based on the final result
plot_cm(masking_policy_gradient,traj_num=100)
