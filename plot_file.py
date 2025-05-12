
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def plot_figures(ex_num, iter_num, modify_list,weight):

    plt.rcParams.update({'font.size': 14})

# ex_num = 0
# iter_num = 5
# modify_list = [112, 113, 114, 115, 68, 69, 70, 71, 56, 57, 58, 59, 24, 25, 26, 27]
# weight = 1e-05

    iteration_list = range(iter_num)

    with open(f'./Data/entropy_values_{ex_num}.pkl', 'rb') as file:
        entropy_list = pickle.load(file)

    with open(f'./Data/x_list_{ex_num}', 'rb') as file:
        x_list = pickle.load(file)





    side_payment_norm_list = []
    for i in iteration_list:
        side_payment_norm_list.append(torch.norm(x_list[i][modify_list]).item())

    total_cost_list = []
    for i in iteration_list:
        total_cost_list.append(entropy_list[i] + weight * torch.sum(x_list[i][modify_list]).item())

    # print(side_payment_norm_list)

    print("The last objective function is", total_cost_list[-1])
    print("The initial entropy value is", entropy_list[0])
    print("The last entropy value is", entropy_list[-1])
    print("The norm of the side payment value is", side_payment_norm_list[-1])

    figure, axis = plt.subplots(3, 1)

    figure.set_size_inches(8, 8)

    # Plot data on the first subplot with a solid red line
    axis[0].plot(iteration_list, total_cost_list, color='green', linestyle='-', label='Objective Function')
    axis[0].set_ylabel("Objective Function")  # Set ylabel for the first subplot
    axis[0].legend()  # Add legend to the first subplot
    axis[0].grid(True)

    # Plot data on the first subplot with a solid red line
    axis[1].plot(iteration_list, entropy_list, color='red', linestyle='-', label='Entropy')
    axis[1].set_ylabel("Estimated Entropy")  # Set ylabel for the first subplot
    axis[1].legend()  # Add legend to the first subplot
    axis[1].grid(True)

    # Plot data on the second subplot with a dashed blue line
    axis[2].plot(iteration_list, side_payment_norm_list, color='blue', linestyle='-', label='Side-Payment')
    axis[2].set_xlabel("Iteration number")  # Set xlabel for the second subplot
    axis[2].set_ylabel("Side-Payment Values")  # Set ylabel for the second subplot
    axis[2].legend()  # Add legend to the second subplot
    axis[2].grid(True)

    # Save and display the plot
    plt.tight_layout()
    plt.savefig(f'./Data/graph_{ex_num}.png')
    plt.show()

