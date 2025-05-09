import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch

plt.rcParams.update({'font.size': 14})

ex_num = 21
iter_num = 200
modify_num = modify_list = list(range(121))
weight = 0

iteration_list = range(iter_num)

with open(f'../Data/entropy_values_{ex_num}.pkl', 'rb') as file:
    entropy_list = pickle.load(file)

with open(f'../Data/x_list_{ex_num}', 'rb') as file:
    x_list = pickle.load(file)

# with open(f'../Data/value_function_list_{ex_num}', 'rb') as file:
#     threshold_list = pickle.load(file)
x = x_list
print(f"Type of x_list: {type(x_list)}")
print(f"Type of x_list[0]: {type(x_list[0])}")
print(f"Type of x_list[0][modify_num]: {type(x_list[0][modify_num])}")
print(f"Shape of x_list[0][modify_num]: {np.shape(x_list[0][modify_num])}")


side_payment_norm_list = []
for i in iteration_list:
    side_payment_norm_list.append(torch.norm(x_list[i][modify_num]).item())

total_cost_list = []
for i in iteration_list:
    total_cost_list.append(entropy_list[i] + weight * torch.sum(x_list[i][modify_num]).item())

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
plt.savefig(f'../Data/graph_{ex_num}.png')
plt.show()


# with open(f'../Data/final_control_policy_{ex_num}.pkl', 'rb') as file:
#     control_policy = pickle.load(file)
#
# print(control_policy)