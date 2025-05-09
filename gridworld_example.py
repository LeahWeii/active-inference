import math

from mdp_env.gridworld_env_multi_init_states import *
# from setup_and_solvers.LP_for_nominal_policy import *
from solver.initial_opacity_gradient_calculation import *
import random
from mdp_env.sensors import *
from mdp_env.hidden_markov_model_of_P2 import *

# logger.add("logs_for_examples/log_file_mario_example_information_theoretic_opacity.log")
#
# logger.info("This is the log file for the 6X6 gridworld with goal states 9, 20, 23 test case.")
ex_num = 36
random.seed(ex_num)
import random

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
    # For modify_list, sample state indices first
    n_actions = 4
    sampled_states = random.sample(range(total_states), n_modify)
    modify_list = []
    for s in sampled_states:
        modify_list.extend([s * n_actions + a for a in range(n_actions)])

    return targets, targets_set_low_reward, targets_set_high_reward,obstacles, initial_state, modify_list


# Initial set-up
penalty_states = []
ncols = 6
nrows = 6
n_targets = 5
n_obstacles = 8
n_modify=4

targets, targets_low_reward,targets_high_reward, obstacles, initial, modify_list = generate_random_indices(ncols, nrows, n_targets, n_obstacles, n_modify)
print(targets, obstacles, initial, modify_list)
reward_states = targets

unsafe_u = []
#or you can specify the environment

initial_dist = dict([])
# considering a single initial state.
for state in range(ncols*nrows):
    if state in initial:
        initial_dist[state] = 1 / len(initial)
    else:
        initial_dist[state] = 0

robot_ts_1 = read_from_file_MDP_old('robotmdp_1.txt')#robot 1 alpha = 0.9
robot_ts_2 = read_from_file_MDP_old('robotmdp_2.txt')#robot 2 alpha = 0.5


# sensor setup
sensors = {'A', 'B', 'C', 'D', 'NO'}


def generate_sensor_areas(grid_size, coverage_percent=90):
    """
    Generate random sensor areas A, B, C, D that cover approximately
    the specified percentage of a gridworld.

    Parameters:
    grid_size: Tuple (width, height) defining the grid dimensions
    coverage_percent: Target percentage of grid to cover (default 90%)

    Returns:
    Dictionary of sets containing coordinates for each sensor area
    """
    width, height = grid_size
    total_cells = width * height

    # Create a set of all grid cells
    all_cells = set((x, y) for x in range(width) for y in range(height))

    # Calculate number of cells to cover
    cells_to_cover = int(total_cells * coverage_percent / 100)

    # Randomly select cells to cover
    cells_to_distribute = random.sample(list(all_cells), cells_to_cover)

    # Divide these cells roughly equally among sensors A, B, C, D
    sets = {"A": set(), "B": set(), "C": set(), "D": set()}

    # Distribute cells to sensors
    for i, cell in enumerate(cells_to_distribute):
        sensor_index = i % 4  # Cycle through sensors
        sensor_key = list(sets.keys())[sensor_index]
        sets[sensor_key].add(cell)

    # Convert to flat indices for easier visualization
    setA = {y * width + x for x, y in sets["A"]}
    setB = {y * width + x for x, y in sets["B"]}
    setC = {y * width + x for x, y in sets["C"]}
    setD = {y * width + x for x, y in sets["D"]}

    # Calculate NO area as the remainder
    setNO = set(range(total_cells)) - (setA | setB | setC | setD)

    # Verify coverage percentage
    # actual_coverage = 100 * (1 - len(setNO) / total_cells)
    # print(f"setA={setA}")

    return setA, setB, setC, setD, setNO

setA, setB, setC, setD, setNO = generate_sensor_areas(grid_size=(ncols,nrows))
print(f'setA={setA}')
print(f'setB={setA}')
print(f'setC={setA}')
print(f'setD={setA}')
print(f'setN0={setA}')

setA ={}

# sensor noise
sensor_noise = 0.1

sensor_net = Sensor()
sensor_net.sensors = sensors

sensor_net.set_coverage('A', setA)
sensor_net.set_coverage('B', setB)
sensor_net.set_coverage('C', setC)
sensor_net.set_coverage('D', setD)
# sensor_net.set_coverage('E', setE)
sensor_net.set_coverage('NO', setNO)

# sensor_net.jamming_actions = masking_action
sensor_net.sensor_noise = sensor_noise
# sensor_net.sensor_cost_dict = sensor_cost

agent_gw_1 = GridworldGui(initial, nrows, ncols, robot_ts_1, targets, obstacles, unsafe_u, initial_dist)
agent_gw_1.mdp.get_supp()
agent_gw_1.mdp.gettrans()
agent_gw_1.mdp.get_reward()
# agent_gw_1.draw_state_labels() #need pygame
trans_1 = agent_gw_1.mdp.trans

agent_gw_2 = GridworldGui(initial, nrows, ncols, robot_ts_2, targets, obstacles, unsafe_u, initial_dist)
agent_gw_2.mdp.get_supp()
agent_gw_2.mdp.gettrans()
agent_gw_2.mdp.get_reward()
# agent_gw_2.draw_state_labels() # need pygame
trans_2 = agent_gw_1.mdp.trans

# reward/ value matrix for each agent.
# value_dict_1 = dict()
# for state in agent_gw_1.mdp.states:
#     if state == 5:
#         value_dict_1[state] = 0.1
#     elif state == 35:
#         value_dict_1[state] = 0.1
#     elif state in penalty_states:
#         value_dict_1[state] = -0.1
#     else:
#         value_dict_1[state] = -0.01
#
# value_dict_2 = dict()
# for state in agent_gw_2.mdp.states:
#     if state == 5:
#         value_dict_2[state] = 0.1
#     elif state == 35:
#         value_dict_2[state] = 0.1
#     elif state in penalty_states:
#         value_dict_2[state] = -20
#     else:
#         value_dict_2[state] = -0.01

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
        value_dict_2[state] = -1



side_payment = {}
for state in agent_gw_1.mdp.states:
    s_idx = agent_gw_1.mdp.states.index(state)
    side_payment[state] = {}
    for action in agent_gw_1.mdp.actlist:
        a_idx = agent_gw_1.mdp.actlist.index(action)
        if s_idx in modify_list:
            side_payment[s_idx][a_idx] = 0
        else:
            side_payment[s_idx][a_idx] = 0
# E_idx = agent_gw_1.actlist.index('E')
# N_idx = agent_gw_1.actlist.index('N')
# s1_idx = agent_gw_1.states.index('')
# idx1 = 4*len(agent_gw_1.actlist) + E_idx
# idx2 = 11*len(agent_gw_1.actlist) + N_idx



# # TODO: The augmented states still consider the gridcells with obstacles. Try by omitting the obstacle filled states
# #  -> reduces computation.
#sp:0-->both 45:E,  0.6-->1S2E
hmm_1 = HiddenMarkovModelP2(agent_gw_1.mdp, sensor_net, side_payment, modify_list, value_dict=value_dict_1)
# policy1 = hmm_1.get_policy_entropy(0.1)
hmm_2 = HiddenMarkovModelP2(agent_gw_2.mdp, sensor_net, side_payment, modify_list, value_dict=value_dict_2)
# policy2 = hmm_2.get_policy_entropy(0.1)
hmm_list = [hmm_1, hmm_2]

# masking_policy_gradient = PrimalDualPolicyGradient(hmm=hmm_p2, iter_num=1000, V=10, T=10, eta=1.5, kappa=0.1, epsilon=threshold)
# masking_policy_gradient.solver()

#for nips, I start ex from 8. 0 is for test(weights=0)
masking_policy_gradient = InitialOpacityPolicyGradient(hmm_list=hmm_list, ex_num=ex_num, weight=0.01,sp = 2,iter_num=200, batch_size=10, V=200,
                                                       T=ncols+ncols,
                                                       eta=0.5) # decreasing eta

masking_policy_gradient.solver()
