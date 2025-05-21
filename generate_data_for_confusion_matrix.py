from solver.initial_opacity_gradient_calculation import *
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_cm(masking_policy_gradient, traj_num=100):
    # Create the folder if it doesn't exist
    output_dir = './Data/confusion_matrix'
    os.makedirs(output_dir, exist_ok=True)

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


            # Convert list to tensor
            probs_no_x = torch.stack(P_T_y_list_no_x)

            # Process with x_opt
            masking_policy_gradient.x = x_opt
            masking_policy_gradient.update_HMMs()
            P_T_y_list_x_opt = masking_policy_gradient.approximate_posterior(y_obs_data)

            # Convert list to tensor
            probs_x_opt = torch.stack(P_T_y_list_x_opt)

            # Check if either contains NaN - if so, skip this sample
            if torch.isnan(probs_no_x).any() or torch.isnan(probs_x_opt).any():
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
        with open('./Data/confusion_matrix/data_no_x_trueType' + str(true_type_num) + f'_{ex_num}.pkl',
                  'wb') as file:
            pickle.dump(data_no_x, file)
        with open('./Data/confusion_matrix/data_x_opt_trueType' + str(true_type_num) + f'_{ex_num}.pkl',
                  'wb') as file:
            pickle.dump(data_x_opt, file)

    # Initialize lists
    true_labels = []
    preds_no_x = []
    preds_x_opt = []

    # Load data
    for true_type in range(num_types):
        with open(f'./Data/confusion_matrix/data_no_x_trueType{true_type}_{ex_num}.pkl', 'rb') as f:
            pred_no_x = pickle.load(f)  # This is a tensor of size traj_num

        with open(f'./Data/confusion_matrix/data_x_opt_trueType{true_type}_{ex_num}.pkl', 'rb') as f:
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
        plt.savefig(f'./Data/confusion_matrix/' + title + f'_{ex_num}.png')


    plot_conf_matrix(cm_no_x, 'Confusion Matrix (no_x)')
    plot_conf_matrix(cm_x_opt, 'Confusion Matrix (x_opt)')
    plt.show()

