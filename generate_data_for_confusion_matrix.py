import math

import torch

from mdp_env.gridworld_env_multi_init_states import *
# from setup_and_solvers.LP_for_nominal_policy import *
from solver.initial_opacity_gradient_calculation import *
import pickle
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



# ex_num = 56
# seed = 56
# random.seed(seed)
# print(f'seed={seed}')
#
#
# # # Initial set-up for 3 types of agent
# # ncols = 6
# # nrows = 6
# # n_targets = 5
# # n_obstacles = 8
# # n_modify=10
#
# # # Initial set-up for 5 types of agent
# # ncols = 8
# # nrows = 8
# # n_targets = 8
# # n_obstacles = 15
# # n_modify=10
#
# # Initial set-up for 5 types of agent(larger)
# ncols = 10
# nrows = 10
# n_targets = 12
# n_obstacles = 20
# n_modify=20
#
#
# def generate_random_indices(ncols, nrows, n_targets, n_obstacles, n_modify):
#     total_states = ncols * nrows
#     all_indices = list(range(total_states))
#
#     # Randomly select target indices
#     targets = random.sample(all_indices, n_targets)
#
#     # Split targets into two non-empty sets
#     split_point = random.randint(1, n_targets - 1)
#     random.shuffle(targets)
#     targets_set_high_reward = targets[:split_point]
#     targets_set_low_reward = targets[split_point:]
#
#     remaining_indices = list(set(all_indices) - set(targets))
#
#     # Randomly select obstacle indices (cannot overlap with targets)
#     obstacles = random.sample(remaining_indices, n_obstacles)
#
#     remaining_indices = list(set(remaining_indices) - set(obstacles))
#
#     # Randomly select initial state (not in targets or obstacles)
#     initial_state = random.choice(remaining_indices)
#     initial_state = {initial_state}
#
#     # For modify_list, sample state indices from non-obstacle states
#     valid_states_for_sampling = list(set(all_indices) - set(obstacles))
#     n_actions = 4
#
#     # Make sure we don't try to sample more states than available
#     n_modify = min(n_modify, len(valid_states_for_sampling))
#
#     sampled_states = random.sample(valid_states_for_sampling, n_modify)
#     modify_list = []
#     for s in sampled_states:
#         modify_list.extend([s * n_actions + a for a in range(n_actions)])
#
#     return targets, targets_set_low_reward, targets_set_high_reward, obstacles, initial_state, modify_list
#
#
#
#
# targets, targets_low_reward, targets_high_reward, obstacles, initial, modify_list = generate_random_indices(ncols, nrows, n_targets, n_obstacles, n_modify)
# # modify_list = list(range(144))
# print(targets, obstacles, initial, modify_list)
# print(f'targets_low_reward={targets_low_reward}')
# print(f'targets_high_reward={targets_high_reward}')
#
#
# unsafe_u = []
# #or you can specify the environment
#
# initial_dist = dict([])
# # considering a single initial state.
# for state in range(ncols*nrows):
#     if state in initial:
#         initial_dist[state] = 1 / len(initial)
#     else:
#         initial_dist[state] = 0
#
#
#
#
# # def generate_sensor_areas(grid_size, coverage_percent=90):
# #     """
# #     Generate random sensor areas A, B, C, D that cover approximately
# #     the specified percentage of a gridworld.
# #
# #     Parameters:
# #     grid_size: Tuple (width, height) defining the grid dimensions
# #     coverage_percent: Target percentage of grid to cover (default 90%)
# #
# #     Returns:
# #     Dictionary of sets containing coordinates for each sensor area
# #     """
# #     width, height = grid_size
# #     total_cells = width * height
# #
# #     # Create a set of all grid cells
# #     all_cells = set((x, y) for x in range(width) for y in range(height))
# #
# #     # Calculate number of cells to cover
# #     cells_to_cover = int(total_cells * coverage_percent / 100)
# #
# #     # Randomly select cells to cover
# #     cells_to_distribute = random.sample(list(all_cells), cells_to_cover)
# #
# #     # Divide these cells roughly equally among sensors A, B, C, D
# #     sets = {"A": set(), "B": set(), "C": set(), "D": set()}
# #
# #     # Distribute cells to sensors
# #     for i, cell in enumerate(cells_to_distribute):
# #         sensor_index = i % 4  # Cycle through sensors
# #         sensor_key = list(sets.keys())[sensor_index]
# #         sets[sensor_key].add(cell)
# #
# #     # Convert to flat indices for easier visualization
# #     setA = {y * width + x for x, y in sets["A"]}
# #     setB = {y * width + x for x, y in sets["B"]}
# #     setC = {y * width + x for x, y in sets["C"]}
# #     setD = {y * width + x for x, y in sets["D"]}
# #
# #     # Calculate NO area as the remainder
# #     setNO = set(range(total_cells)) - (setA | setB | setC | setD)
# #
# #     return setA, setB, setC, setD, setNO
# #
# # setA, setB, setC, setD, setNO = generate_sensor_areas(grid_size=(ncols,nrows))
#
#
# import random
#
# import random
#
# def generate_sensor_areas(grid_size, coverage_percent=90, num_sensors=10):
#     """
#     Generate random sensor areas (e.g., A to J) that cover approximately
#     the specified percentage of a gridworld.
#
#     Parameters:
#     grid_size: Tuple (width, height) defining the grid dimensions
#     coverage_percent: Target percentage of grid to cover (default 90%)
#     num_sensors: Number of distinct sensors (default 10)
#
#     Returns:
#     Tuple: sensor_sets (list of sets for each sensor) + NO area set
#     """
#     width, height = grid_size
#     total_cells = width * height
#
#     # Create a set of all grid cells
#     all_cells = set((x, y) for x in range(width) for y in range(height))
#
#     # Calculate number of cells to cover
#     cells_to_cover = int(total_cells * coverage_percent / 100)
#
#     # Randomly select cells to cover
#     cells_to_distribute = random.sample(list(all_cells), cells_to_cover)
#
#     # Generate sensor labels (e.g., 'A', 'B', ..., up to num_sensors)
#     sensor_labels = [chr(ord('A') + i) for i in range(num_sensors)]
#     sets = {label: set() for label in sensor_labels}
#
#     # Distribute cells to sensors
#     for i, cell in enumerate(cells_to_distribute):
#         sensor_label = sensor_labels[i % num_sensors]
#         sets[sensor_label].add(cell)
#
#     # Convert to flat indices
#     sensor_sets_flat = [
#         {y * width + x for x, y in sets[label]} for label in sensor_labels
#     ]
#
#     # Calculate NO area
#     all_sensor_cells = set().union(*sensor_sets_flat)
#     setNO = set(range(total_cells)) - all_sensor_cells
#
#     # Return all sensor sets + the NO set
#     return tuple(sensor_sets_flat) + (setNO,)
#
#
#
# # Configuration
# sensor_noise = 0.05
# print(f'sensor_noise = {sensor_noise}')
#
# num_sensors = 20  # or any other number of sensors you choose
# sensor_labels = [chr(ord('A') + i) for i in range(num_sensors)]
#
# # Generate sensor areas
# sensor_sets = generate_sensor_areas((ncols, nrows), coverage_percent=95, num_sensors=num_sensors)
# *sensor_coverage_list, setNO = sensor_sets
#
# # Print each sensor's coverage
# for label, coverage in zip(sensor_labels, sensor_coverage_list):
#     print(f"set{label} = {coverage}")
#
# # Print NO set
# print(f"setNO = {setNO}")
#
# # Initialize sensor coverage dictionary
# sensor_coverage_dict = {label: sensor_coverage_list[i] for i, label in enumerate(sensor_labels)}
# sensor_coverage_dict["NO"] = setNO
#
# # Sensor setup
# sensor_net = Sensor()
# sensor_net.sensors = set(sensor_coverage_dict.keys())
#
# # Set coverage for all sensors
# for label, coverage in sensor_coverage_dict.items():
#     sensor_net.set_coverage(label, coverage)
#
#
#
# #initialize differenrt types of agents with different transition and reward, but same initial state and environment
# robot_ts_1 = read_from_file_MDP_old('./robotmdp_para/robotmdp_1.txt')#robot 1 alpha = 0.9
# robot_ts_2 = read_from_file_MDP_old('./robotmdp_para/robotmdp_2.txt')#robot 2 alpha = 0.6
# robot_ts_3 = read_from_file_MDP_old('./robotmdp_para/robotmdp_3.txt')#robot 2 alpha = 0.9
# robot_ts_4 = read_from_file_MDP_old('./robotmdp_para/robotmdp_4.txt')#robot 2 alpha = 0.7
# robot_ts_5 = read_from_file_MDP_old('./robotmdp_para/robotmdp_5.txt')#robot 2 alpha = 1
#
# agent_gw_1 = GridworldGui(initial, nrows, ncols, robot_ts_1, targets, obstacles, unsafe_u, initial_dist)
# agent_gw_1.mdp.get_supp()
# agent_gw_1.mdp.gettrans()
# agent_gw_1.mdp.get_reward()
# t1 = agent_gw_1.mdp.trans[27]
#
#
# agent_gw_2 = GridworldGui(initial, nrows, ncols, robot_ts_2, targets, obstacles, unsafe_u, initial_dist)
# agent_gw_2.mdp.get_supp()
# agent_gw_2.mdp.gettrans()
# agent_gw_2.mdp.get_reward()
#
# agent_gw_3 = GridworldGui(initial, nrows, ncols, robot_ts_3, targets, obstacles, unsafe_u, initial_dist)
# agent_gw_3.mdp.get_supp()
# agent_gw_3.mdp.gettrans()
# agent_gw_3.mdp.get_reward()
#
# agent_gw_4 = GridworldGui(initial, nrows, ncols, robot_ts_4, targets, obstacles, unsafe_u, initial_dist)
# agent_gw_4.mdp.get_supp()
# agent_gw_4.mdp.gettrans()
# agent_gw_4.mdp.get_reward()
#
# agent_gw_5 = GridworldGui(initial, nrows, ncols, robot_ts_5, targets, obstacles, unsafe_u, initial_dist)
# agent_gw_5.mdp.get_supp()
# agent_gw_5.mdp.gettrans()
# agent_gw_5.mdp.get_reward()
#
#
# value_dict_1 = dict()
# for state in agent_gw_1.mdp.states:
#     if state in targets_low_reward:
#         value_dict_1[state] = 1
#     elif state in targets_high_reward:
#         value_dict_1[state] = 2
#     else:
#         value_dict_1[state] = -0.5
#
# value_dict_2 = dict()
# for state in agent_gw_2.mdp.states:
#     if state in targets_low_reward:
#         value_dict_2[state] = 1
#     elif state in targets_high_reward:
#         value_dict_2[state] = 2
#     else:
#         value_dict_2[state] = -2
#
# value_dict_3 = dict()
# for state in agent_gw_3.mdp.states:
#     if state in targets_low_reward:
#         value_dict_3[state] = 0.1
#     elif state in targets_high_reward:
#         value_dict_3[state] = 4
#     else:
#         value_dict_3[state] = 0
#
# value_dict_4 = dict()
# for state in agent_gw_4.mdp.states:
#     if state in targets_low_reward:
#         value_dict_4[state] = 1
#     elif state in targets_high_reward:
#         value_dict_4[state] = 1
#     else:
#         value_dict_4[state] = -1
#
# value_dict_5 = dict()
# for state in agent_gw_5.mdp.states:
#     if state in targets_low_reward:
#         value_dict_5[state] = 1
#     elif state in targets_high_reward:
#         value_dict_5[state] = 1
#     else:
#         value_dict_5[state] = -0.1
#
#
#
# side_payment = {}
# for state in agent_gw_1.mdp.states:
#     s_idx = agent_gw_1.mdp.states.index(state)
#     side_payment[state] = {}
#     for action in agent_gw_1.mdp.actlist:
#         a_idx = agent_gw_1.mdp.actlist.index(action)
#         side_payment[s_idx][a_idx] = 0
# # E_idx = agent_gw_1.actlist.index('E')
# # N_idx = agent_gw_1.actlist.index('N')
# # s1_idx = agent_gw_1.states.index('')
# # idx1 = 4*len(agent_gw_1.actlist) + E_idx
# # idx2 = 11*len(agent_gw_1.actlist) + N_idx
#
#
#
# # # TODO: The augmented states still consider the gridcells with obstacles. Try by omitting the obstacle filled states
# # #  -> reduces computation.
# #sp:0-->both 45:E,  0.6-->1S2E
# hmm_1 = HiddenMarkovModelP2(agent_gw_1.mdp, sensor_net, side_payment, modify_list, value_dict=value_dict_1)
# hmm_2 = HiddenMarkovModelP2(agent_gw_2.mdp, sensor_net, side_payment, modify_list, value_dict=value_dict_2)
# hmm_3 = HiddenMarkovModelP2(agent_gw_3.mdp, sensor_net, side_payment, modify_list, value_dict=value_dict_3)
# hmm_4 = HiddenMarkovModelP2(agent_gw_4.mdp, sensor_net, side_payment, modify_list, value_dict=value_dict_4)
# hmm_5 = HiddenMarkovModelP2(agent_gw_5.mdp, sensor_net, side_payment, modify_list, value_dict=value_dict_5)
# hmm_list = [hmm_1, hmm_2, hmm_3, hmm_4, hmm_5]
# # hmm_list = [hmm_1, hmm_2, hmm_3]
#
# # masking_policy_gradient = PrimalDualPolicyGradient(hmm=hmm_p2, iter_num=1000, V=10, T=10, eta=1.5, kappa=0.1, epsilon=threshold)
# # masking_policy_gradient.solver()
#
# #for nips, I start ex from 8. 0 is for test(weights=0)
# masking_policy_gradient = InitialOpacityPolicyGradient(hmm_list=hmm_list, ex_num=ex_num, weight=1e-01, sp = 0, iter_num=200, batch_size=10, V=200,
#                                                        T=ncols+nrows)


def plot_cm(masking_policy_gradient, traj_num=100):
    ex_num = masking_policy_gradient.ex_num
    with open(f'./Data/x_list_{ex_num}', 'rb') as file:
        x_list = pickle.load(file)

    x_opt = x_list[-1]

    no_x = torch.zeros_like(x_opt)
    num_types = len(masking_policy_gradient.hmm_list)
    count = 0
    for true_type_num in range(num_types):
        count += 1
        print(count)
        data_no_x = torch.zeros(traj_num)
        data_x_opt = torch.zeros(traj_num)
        valid_samples = 0

        while valid_samples < traj_num:
            state_data, action_data, y_obs_data = masking_policy_gradient.sample_trajectories(true_type_num)

            # Process with no_x
            masking_policy_gradient.x = no_x
            masking_policy_gradient.update_HMMs()
            P_T_y_list_no_x = masking_policy_gradient.approximate_posterior(y_obs_data)
            print(f'no_x={P_T_y_list_no_x}')

            # Convert list to tensor
            probs_no_x = torch.stack(P_T_y_list_no_x)

            # Process with x_opt
            masking_policy_gradient.x = x_opt
            masking_policy_gradient.update_HMMs()
            P_T_y_list_x_opt = masking_policy_gradient.approximate_posterior(y_obs_data)
            print(f'x_opt={P_T_y_list_x_opt}')

            # Convert list to tensor
            probs_x_opt = torch.stack(P_T_y_list_x_opt)

            # Check if either contains NaN - if so, skip this sample
            if torch.isnan(probs_no_x).any() or torch.isnan(probs_x_opt).any():
                print("Skipping sample with NaN values")
                continue

            # If we reach here, the sample is valid
            # Sample from the distributions
            distribution_no_x = torch.distributions.Categorical(probs=probs_no_x)
            data_no_x[valid_samples] = distribution_no_x.sample()

            distribution_x_opt = torch.distributions.Categorical(probs=probs_x_opt)
            data_x_opt[valid_samples] = distribution_x_opt.sample()

            valid_samples += 1

            if valid_samples % 10 == 0:
                print(f"Collected {valid_samples}/{traj_num} valid samples for type {true_type_num}")

        # Save the collected data
        with open('./Data/data_for_confusion_matrix/data_no_x_trueType' + str(true_type_num) + f'_{ex_num}.pkl',
                  'wb') as file:
            pickle.dump(data_no_x, file)
        with open('./Data/data_for_confusion_matrix/data_x_opt_trueType' + str(true_type_num) + f'_{ex_num}.pkl',
                  'wb') as file:
            pickle.dump(data_x_opt, file)

    # Initialize lists
    true_labels = []
    preds_no_x = []
    preds_x_opt = []

    # Load data
    for true_type in range(num_types):
        with open(f'./Data/data_for_confusion_matrix/data_no_x_trueType{true_type}_{ex_num}.pkl', 'rb') as f:
            pred_no_x = pickle.load(f)  # This is a tensor of size traj_num

        with open(f'./Data/data_for_confusion_matrix/data_x_opt_trueType{true_type}_{ex_num}.pkl', 'rb') as f:
            pred_x_opt = pickle.load(f)

        # Add true_type label for each trajectory
        true_labels.extend([true_type] * traj_num)

        # Convert tensors to numpy arrays and extend the lists
        preds_no_x.extend(pred_no_x.numpy())
        preds_x_opt.extend(pred_x_opt.numpy())

    # Now arrays should be of consistent length
    true_labels = np.array(true_labels)
    preds_no_x = np.array(preds_no_x)
    preds_x_opt = np.array(preds_x_opt)

    # Now you can calculate the confusion matrix
    cm_no_x = confusion_matrix(true_labels, preds_no_x, labels=range(num_types))
    cm_x_opt = confusion_matrix(true_labels, preds_x_opt, labels=range(num_types))


    def plot_conf_matrix(cm, title):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(num_types),
                    yticklabels=range(num_types))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        # plt.title(title)
        plt.tight_layout()
        plt.savefig(f'./Data/data_for_confusion_matrix/' + title + f'_{ex_num}.png')


    plot_conf_matrix(cm_no_x, 'Confusion Matrix (no_x)')
    plot_conf_matrix(cm_x_opt, 'Confusion Matrix (x_opt)')
    plt.show()

