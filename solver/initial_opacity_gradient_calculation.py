import matplotlib.pyplot as plt
from mdp_env.hidden_markov_model_of_P2 import *
from plot_file import *
import torch
import time
import torch.nn.functional as F
import itertools
import pickle
import math
from contextlib import contextmanager

@contextmanager
def timing(name):
    start = time.time()
    yield
    end = time.time()
    print(f"[{name}] took {end - start:.4f} seconds")



# Check and set device
device = torch.device("cpu")
print(f"Using device: {device}")

# Set memory management options for GPU
if torch.cuda.is_available():
    # Enable TF32 precision for faster computation on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Set to benchmark mode for optimized computation
    torch.backends.cudnn.benchmark = True


class InitialOpacityPolicyGradient:
    def __init__(self, hmm_list, ex_num,weight, sp, true_type_num=1, iter_num=1000, batch_size=1, V=100, T=12):
        print(f"ex_num={ex_num}")
        for hmm in hmm_list:
            if not isinstance(hmm, HiddenMarkovModelP2):
                raise TypeError("Expected hmm to be an instance of HiddenMarkovModelP2.")

        self.num_of_types = len(hmm_list)
        print(f'num of types={self.num_of_types}')
        self.true_type_num = true_type_num
        self.modify_list = hmm_list[0].modify_list

        # Move prior to device immediately
        self.prior = torch.ones(self.num_of_types, device=device) / self.num_of_types

        self.hmm_list = hmm_list  # Hidden markov model of type 1.
        self.iter_num = iter_num  # number of iterations for gradient ascent
        print(f'iter_num={self.iter_num}')
        self.ex_num = ex_num
        self.V = V  # number of sampled trajectories.
        self.batch_size = batch_size  # number of trajectories processed in each batch.
        self.T = T  # length of the sampled trajectory.  # step size for theta.
        print(f'batch_size={self.batch_size}')
        print(f'V={self.V}')
        # The states and actions of original MDP
        self.states = self.hmm_list[0].states
        self.actions = self.hmm_list[0].actions
        self.num_of_states = len(self.states)
        self.num_of_actions = len(self.actions)

        # About side payment - create directly on GPU
        self.x_size = self.num_of_states * self.num_of_actions
        self.x = torch.zeros(self.x_size, dtype=torch.float32, device=device, requires_grad=False)
        self.x[self.modify_list] = sp
        print(f'initial sp in modify list={self.x[self.modify_list][0] }')
        self.weight = weight
        print(f'weight={self.weight}')

        # Initialize lists
        self.theta_torch_list = []
        self.transition_mat_torch_list = []
        self.mu_0_torch_list = []
        self.B_torch_list = []
        self.T_theta_list = []

        # Get all the lists we need
        self.get_all_lists()

        # Results tracking
        self.entropy_list = []
        self.total_cost_list = []
        self.iteration_list = []
        self.theta_torch_collection = []
        self.x_list = []

    def get_all_lists(self):
        for type_num in range(self.num_of_types):
            hmm = self.hmm_list[type_num]

            # Move all data to device directly when creating
            # Construct theta in pyTorch ways
            theta_torch = torch.tensor(hmm.optimal_theta, dtype=torch.float32, device=device)
            theta_torch.requires_grad_(True)
            self.theta_torch_list.append(theta_torch)

            # Transition matrix
            transition_mat_torch = torch.tensor(hmm.transition_mat, dtype=torch.float32, device=device)
            self.transition_mat_torch_list.append(transition_mat_torch)

            # Initial distribution
            mu_0_torch = torch.tensor(hmm.mu_0, dtype=torch.float32, device=device)
            self.mu_0_torch_list.append(mu_0_torch)

            # Construct the transition matrices
            self.T_theta_list.append(self.construct_transition_matrix_T_theta_torch(type_num))

            # Construct observation matrices
            self.B_torch_list.append(self.construct_B_matrix_torch(type_num))

    def update_the_lists(self):
        self.theta_torch_list = []
        self.T_theta_list = []
        for type_num in range(self.num_of_types):
            hmm = self.hmm_list[type_num]

            # Move data to device
            theta_torch = torch.tensor(hmm.optimal_theta, dtype=torch.float32, device=device)
            theta_torch.requires_grad_(True)
            self.theta_torch_list.append(theta_torch)

            # Update transition matrices
            self.T_theta_list.append(self.construct_transition_matrix_T_theta_torch(type_num))

    def convert_policy(self, policy):
        # Create tensor directly on GPU
        policy_tensor = torch.zeros(self.x_size, dtype=torch.float32, device=device)

        # Pre-compute indices for faster access
        indices = [(st, act) for st in self.states for act in self.actions]

        # Vectorized assignment where possible
        for i, (st, act) in enumerate(indices):
            policy_tensor[i] = policy[st][act]

        return policy_tensor

    def construct_value_matrix(self, type_num):
        hmm = self.hmm_list[type_num]
        value_matrix = torch.zeros(len(hmm.states), len(hmm.actions), device=device)
        for s in hmm.value_dict:
            for a in hmm.value_dict[s]:
                value_matrix[s, a] = hmm.value_dict[s][a]
        return value_matrix

    def sample_action_torch(self, state, type_num):
        hmm = self.hmm_list[type_num]
        theta = self.theta_torch_list[type_num]

        # Sample action given state and theta, following softmax policy
        state_indx = hmm.states_indx_dict[state]

        # Extract logits corresponding to the given state
        logits = theta[state_indx]
        logits = logits - logits.max()  # Logit regularization

        # Compute softmax probabilities
        action_probs = F.softmax(logits, dim=0)

        # Sample action based on probabilities
        action = torch.multinomial(action_probs, num_samples=1).item()
        return action

    def sample_trajectories(self, type_num):
        hmm = self.hmm_list[type_num]

        # Create tensors directly on GPU
        state_data = torch.zeros(self.batch_size, self.T, dtype=torch.int32, device=device)
        action_data = torch.zeros(self.batch_size, self.T, dtype=torch.int32, device=device)

        # Create a tensor for observations rather than a list
        # (if possible, depending on the structure of observations)
        y_obs_data = []  # This might need to remain a list due to variable length observations

        # Process batches in parallel if possible
        for v in range(self.batch_size):
            y = []
            # Get initial state
            initial_states = list(hmm.initial_states)
            state_idx = torch.randint(0, len(initial_states), (1,), device=device).item()
            state = initial_states[state_idx]

            act = self.sample_action_torch(state, type_num)

            for t in range(self.T):
                # This still requires CPU operation
                y.append(hmm.sample_observation_same_NO_Null(state))

                # Store in GPU tensor directly
                s = hmm.states_indx_dict[state]
                state_data[v, t] = s
                action_data[v, t] = act

                # These operations still happen on CPU
                state = hmm.sample_next_state(state, act)
                act = self.sample_action_torch(state, type_num)

            y_obs_data.append(y)

        return state_data, action_data, y_obs_data

    def construct_transition_matrix_T_theta_torch(self, type_num):
        transition_mat_torch = self.transition_mat_torch_list[type_num]
        theta_torch = self.theta_torch_list[type_num]

        # Apply softmax to get policy probabilities
        logits = theta_torch.clone()
        logits = logits - logits.max()  # Regularize

        pi_theta = F.softmax(logits, dim=1)

        # Compute T_theta using einsum for efficiency
        T_theta = torch.einsum('sa, sna->ns', pi_theta, transition_mat_torch)

        return T_theta

    def construct_B_matrix_torch(self, type_num):
        hmm = self.hmm_list[type_num]
        # Create emission probability matrix
        B_torch = torch.zeros(len(hmm.observations), len(hmm.states), device=device)

        for state, obs in itertools.product(hmm.states, hmm.observations):
            B_torch[hmm.observations_indx_dict[obs], hmm.states_indx_dict[state]] = \
                hmm.emission_prob[state][obs]

        return B_torch

    def construct_A_matrix_torch(self, type_num, o_t):
        hmm = self.hmm_list[type_num]
        B_torch = self.B_torch_list[type_num]
        T_theta = self.T_theta_list[type_num]

        # Get observation index
        o_t_index = hmm.observations_indx_dict[o_t]

        # Create diagonal matrix
        B_diag = torch.diag(B_torch[o_t_index, :])

        # Matrix multiplication
        return T_theta @ B_diag

    def compute_A_matrices(self, type_num, y_v):
        # Build A matrices for observation sequence
        A_matrices = []
        for o_t in y_v:
            A_o_t = self.construct_A_matrix_torch(type_num, o_t)
            A_matrices.append(A_o_t)
        return A_matrices

    def compute_probability_of_observations_given_type(self, type_num, A_matrices):
        theta_torch = self.theta_torch_list[type_num]

        # Start with initial distribution
        result_prob = self.mu_0_torch_list[type_num]

        # Multiply through A matrices
        for A in A_matrices:
            result_prob = torch.matmul(A, result_prob)

        # Sum for final probability
        result_prob_P_y_g_T = result_prob.sum()

        # Compute gradient using autograd
        result_prob_P_y_g_T.backward(retain_graph=True)
        gradient_P_y_g_T = theta_torch.grad.clone()

        # Clear gradients
        theta_torch.grad.zero_()

        return result_prob_P_y_g_T, gradient_P_y_g_T

    def compute_probability_of_observations(self, A_matrices_list):
        result_prob_P_y = torch.tensor(0.0, device=device)
        gradient_P_y = torch.zeros([self.num_of_states, self.num_of_actions], device=device)

        for type_num in range(self.num_of_types):
            result_prob_P_y_g_T, gradient_P_y_g_T = self.compute_probability_of_observations_given_type(
                type_num, A_matrices_list[type_num])

            result_prob_P_y = result_prob_P_y + result_prob_P_y_g_T * self.prior[type_num]
            gradient_P_y = gradient_P_y + self.prior[type_num] * gradient_P_y_g_T

        return result_prob_P_y, gradient_P_y

    def P_T_g_Y(self, type_num, A_matrices_list):
        # Compute posterior probability
        prob_P_y_T, gradient_P_y_T = self.compute_probability_of_observations_given_type(type_num,
                                                                                         A_matrices_list[type_num])
        prob_P_y, gradient_P_y = self.compute_probability_of_observations(A_matrices_list)

        # Calculate P(T|Y)
        P_T_y = prob_P_y_T * self.prior[type_num] / prob_P_y

        # Calculate gradient
        gradient_P_T_y = ((self.prior[type_num] / prob_P_y) * gradient_P_y_T -
                          (self.prior[type_num] * prob_P_y_T / prob_P_y ** 2) * gradient_P_y)

        return P_T_y, gradient_P_T_y, prob_P_y, gradient_P_y

    def approximate_posterior(self, y_obs_data):
        # Initialize with zeros on device
        P_T_y_list = [torch.tensor(0.0, device=device) for _ in range(self.num_of_types)]

        for t in range(self.num_of_types):
            for v in range(self.batch_size):
                y_v = y_obs_data[v]

                # Prepare A matrices for all types
                A_matrices_list = []
                for type_num in range(self.num_of_types):
                    A_matrices_list.append(self.compute_A_matrices(type_num, y_v))

                # Calculate posterior
                P_T_y, _, _, _ = self.P_T_g_Y(t, A_matrices_list)

                # Clamp to prevent numerical issues
                P_T_y = torch.clamp(P_T_y, min=0.0, max=1.0)

                P_T_y_list[t] = P_T_y_list[t] + P_T_y.item()

            # Average over batch
            P_T_y_list[t] = P_T_y_list[t] / self.batch_size

        return P_T_y_list

    def approximate_conditional_entropy_and_gradient_S0_given_Y(self, y_obs_data):
        """Optimized entropy and gradient calculation for GPU"""
        # Initialize on device
        H = torch.tensor(0.0, dtype=torch.float32, device=device)
        nabla_H = torch.zeros([self.num_of_types, self.num_of_states, self.num_of_actions], device=device)

        # Pre-compute log2 constant
        log2_base = torch.log(torch.tensor(2.0, device=device))

        # Process each batch item
        for v in range(self.batch_size):
            y_v = y_obs_data[v]

            # Prepare A matrices for all types
            A_matrices_list = []
            for type_num in range(self.num_of_types):
                A_matrices_list.append(self.compute_A_matrices(type_num, y_v))

            # Process each type
            for type_num in range(self.num_of_types):
                # Calculate posterior and related values - don't use torch.no_grad() here
                P_T_y, gradient_P_T_y, result_P_y, gradient_P_y = self.P_T_g_Y(type_num, A_matrices_list)
                # print(f"type_num, P_T_y={type_num,P_T_y}")

                # Clamp to prevent numerical issues - use a small positive value instead of 0
                P_T_y = torch.clamp(P_T_y, min=1e-10, max=1.0)

                # Calculate log2 safely using torch.log and division (more GPU efficient)
                log2_P_T_y = torch.log(P_T_y) / log2_base

                # Calculate contribution to entropy
                term_p_logp = P_T_y * log2_P_T_y
                # print(f'term_p_logp={term_p_logp}')

                # Calculate gradient term as before, but using the log base constant
                gradient_term = (log2_P_T_y * gradient_P_T_y) + (
                        P_T_y * log2_P_T_y * gradient_P_y / result_P_y) + (
                                        gradient_P_T_y / log2_base)

                # Update accumulators
                H = H + term_p_logp
                nabla_H[type_num, :, :] = nabla_H[type_num, :, :] + gradient_term

        # Average over batch
        H = H / self.batch_size
        nabla_H = nabla_H / self.batch_size

        return -H, -nabla_H

    def dtheta_T_dx(self, type_num):
        """Highly optimized version of dtheta_T_dx for GPU"""
        # Precompute indices to avoid repeated list processing
        modify_indices = torch.tensor(self.modify_list, dtype=torch.long, device=device)

        # Pre-allocate the result tensor
        grad = torch.zeros((self.x_size, self.x_size), dtype=torch.float32, device=device)

        # Process all indices in the modify_list in a batched way if possible
        def process_batch(indices_batch):
            # Process multiple indices in parallel
            results = []
            for idx in indices_batch:
                results.append(self.dtheta_T_dx_line(idx.item(), type_num))
            return results

        # Process in batches of 10 (or another appropriate batch size)
        batch_size = 10
        for i in range(0, len(modify_indices), batch_size):
            batch_indices = modify_indices[i:i + batch_size]
            results = process_batch(batch_indices)

            # Assign results
            for j, result in enumerate(results):
                idx = batch_indices[j].item()
                grad[:, idx] = result



        return grad

    def dtheta_T_dx_line(self, index, type_num, epsilon=0.0001, max_iterations=100):
        """Highly optimized version of dtheta_T_dx_line for GPU"""
        hmm = self.hmm_list[type_num]

        # Cache policy tensors
        if not hasattr(self, 'policy_tensors'):
            self.policy_tensors = {}

        if type_num not in self.policy_tensors:
            # policy_m = self.convert_policy(hmm.policy)
            self.policy_tensors[type_num] = self.convert_policy(hmm.policy)

        policy_tensor = self.policy_tensors[type_num]

        # Cache P matrices
        if not hasattr(self, 'P_matrices'):
            self.P_matrices = {}

        if type_num not in self.P_matrices:
            P_np = self.construct_P(type_num)
            if isinstance(P_np, torch.Tensor):
                self.P_matrices[type_num] = P_np.detach().clone().to(device=device)
            else:
                self.P_matrices[type_num] = torch.tensor(P_np, dtype=torch.float32, device=device)

        P_matrix = self.P_matrices[type_num]

        # Initialize with zeros directly on GPU
        dtheta = torch.zeros(self.x_size, dtype=torch.float32, device=device)
        r_indicator = torch.zeros(self.x_size, dtype=torch.float32, device=device)
        r_indicator[index] = 1.0

        # Use a constant tensor for discount factor
        disc_factor = torch.tensor(hmm.agent_mdp.disc_factor, dtype=torch.float32, device=device)

        # Use a power iteration method for faster convergence
        # This can be much faster than the standard iteration
        for iteration in range(max_iterations):
            dtheta_old = dtheta.clone()

            # Optimize matrix multiplication
            policy_dtheta = policy_tensor * dtheta
            transition_effect = torch.matmul(P_matrix, policy_dtheta)
            dtheta = r_indicator + disc_factor * transition_effect

            # Check convergence with early stopping
            delta = torch.max(torch.abs(dtheta - dtheta_old)).item()
            if delta <= epsilon:
                break

        return dtheta

    def construct_P(self, type_num):
        hmm = self.hmm_list[type_num]

        # Create tensor directly on GPU
        P = torch.zeros((self.x_size, self.x_size), dtype=torch.float32, device=device)

        # Pre-calculate indices to avoid repeated computations
        action_range = torch.arange(self.num_of_actions, device=device)

        for i in range(self.num_of_states):
            for j in range(self.num_of_actions):
                # Get action from index
                action = hmm.actions[j]

                # Process transitions for this state-action pair
                for next_index, prob in hmm.transition_dict[i][action].items():
                    if hmm.states[next_index] != 'Sink':
                        # Calculate row index
                        row_idx = i * self.num_of_actions + j

                        # Calculate column indices
                        start_col_idx = next_index * self.num_of_actions
                        col_indices = start_col_idx + action_range

                        # Set values in the P matrix
                        P[row_idx, col_indices] = prob

        return P

    def dtheta_dx(self):
        """Highly optimized version of dtheta_dx for GPU"""
        # Create a tensor to hold all results
        result_shape = (self.num_of_types * self.x_size, self.x_size)
        all_grads = torch.zeros(result_shape, dtype=torch.float32, device=device)

        # Pre-compute the mask once
        mask = torch.zeros(self.x_size, dtype=torch.float32, device=device)
        mask[self.modify_list] = 1

        # Process each type - potential for parallelization
        for type_num in range(self.num_of_types):
            # Calculate start index for this type
            start_idx = type_num * self.x_size

            # Get gradient tensor
            with torch.no_grad():
                # This is the most expensive operation - optimize dtheta_T_dx further
                grad_T = self.dtheta_T_dx(type_num)


                # Convert to tensor only if needed
                if not isinstance(grad_T, torch.Tensor):
                    grad_T = torch.tensor(grad_T, dtype=torch.float32, device=device)

                # Apply mask in a single broadcast operation
                grad_T = grad_T * mask.unsqueeze(0) if len(grad_T.shape) > 1 else grad_T * mask

                # Assign to result tensor without reshaping
                if len(grad_T.shape) == 2:
                    all_grads[start_idx:start_idx + self.x_size] = grad_T
                else:
                    all_grads[start_idx:start_idx + self.x_size] = grad_T.unsqueeze(0)

        return all_grads

    def dh_dx(self):
        """GPU-optimized version of dh_dx with vectorized operations"""
        # Create tensor directly on GPU with zeros
        grad = torch.zeros(self.x_size, dtype=torch.float32, device=device)

        # Convert modify_list to tensor if it's not already
        modify_indices = torch.tensor(self.modify_list, dtype=torch.long, device=device)

        # Get x values for modify_list indices
        x_values = self.x[modify_indices]

        # Create a sign tensor based on x values (1 for positive, -1 for negative)
        signs = torch.where(x_values >= 0,
                            torch.ones_like(x_values, device=device),
                            -torch.ones_like(x_values, device=device))

        # Multiply by weight
        weight_values = signs * self.weight

        # Scatter the weight values to the grad tensor at modify_list indices
        grad.scatter_(0, modify_indices, weight_values)

        return grad

    def total_derivative(self, y_obs_data):
        # Calculate entropy and gradient
        # with timing("approximate_conditional_entropy_calculation"):
        H, nabla_H = self.approximate_conditional_entropy_and_gradient_S0_given_Y(y_obs_data)

        # Reshape gradient
        nabla_H = nabla_H.reshape(-1)
        nabla_H = nabla_H.unsqueeze(0)

        # Get jacobian of theta with respect to x
        # with timing("dtheta_dx_calculation"):
        nabla_Q = self.dtheta_dx()
            # nabla_Q = torch.tensor(nabla_Q, dtype=torch.float32, device=device)

        # Calculate product
        # with timing("matrix_multiplication"):
        product = torch.matmul(nabla_H, nabla_Q)
        # torch.set_printoptions(precision=10, sci_mode=True, threshold=float('inf'))
        # print(f"product = {product}")
        product = product + self.dh_dx()

        return H, product

    def get_side_payment(self, x):
        side_payment = {}

        # Create mask on device
        device = x.device
        mask = ~torch.isin(
            torch.arange(len(x), device=device),
            torch.tensor(self.modify_list, device=device)
        )

        # Apply mask
        x = torch.where(mask, torch.zeros_like(x), x)

        # Convert to dictionary format
        idx = 0
        for state in self.states:
            s_idx = self.states.index(state)
            side_payment[state] = {}
            for action in self.actions:
                a_idx = self.actions.index(action)
                side_payment[s_idx][a_idx] = x[idx].item()
                idx += 1

        return side_payment

    def update_HMMs(self):
        for type_num in range(self.num_of_types):
            # Update side payment
            self.hmm_list[type_num].side_payment = self.get_side_payment(self.x)

            # Update value dictionary
            self.hmm_list[type_num].get_value_dict()

            # Update policy
            self.hmm_list[type_num].optimal_V, self.hmm_list[type_num].policy = self.hmm_list[
                type_num].get_policy_entropy(tau=0.1)

            # Update theta
            self.hmm_list[type_num].optimal_theta = self.hmm_list[type_num].get_optimal_theta(
                self.hmm_list[type_num].optimal_V)

            # Update lists
            self.update_the_lists()

    def solver(self):
        print('eta = 1*math.exp(-0.005 * i)')
        torch.set_printoptions(precision=10, sci_mode=True)

        # Pre-allocate tensors for reuse
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Set higher priority for this process
            torch.cuda.set_device(0)  # Use primary GPU

        # Batch more operations together
        for i in range(self.iter_num):
            start = time.time()

            # Initialize
            approximate_cond_entropy = torch.tensor(0.0, device=device)
            # Make sure grad has the same shape as what total_derivative returns
            # Initially set it to None and initialize it with the first result
            grad = None

            # Calculate trajectory iterations
            trajectory_iter = int(self.V / self.batch_size)

            # Update HMMs - this is likely a CPU-heavy operation
            # with timing("update_HMMs"):
            self.update_HMMs()

            # Save current theta and x - minimize list operations during timing
            # with timing("save_state"):
            self.theta_torch_collection.append([t.clone().detach().cpu() for t in self.theta_torch_list])
            self.x_list.append(self.x.clone().detach().cpu())

            # Update learning rate with decay
            eta = 1*math.exp(-0.005 * i)


            # Prepare all trajectories at once if possible
            # with timing("generate_trajectories"):
            all_trajectory_data = []
            for j in range(trajectory_iter):
                with torch.no_grad():
                    state_data, action_data, y_obs_data = self.sample_trajectories(self.true_type_num)
                    all_trajectory_data.append((state_data, action_data, y_obs_data))

            # Process all trajectories
            # with timing("process_trajectories"):
            for idx, (state_data, action_data, y_obs_data) in enumerate(all_trajectory_data):
                # Calculate derivatives
                # with timing("total_derivative"):
                H, grad_new = self.total_derivative(y_obs_data)
                approximate_cond_entropy += H.item()

                # Initialize grad with the first result or add to existing
                if grad is None:
                    grad = grad_new.detach()  # Detach from computation graph
                else:
                    # Make sure dimensions match before adding
                    if grad.shape != grad_new.shape:
                        # Reshape if needed
                        if len(grad.shape) != len(grad_new.shape):
                            if len(grad.shape) == 1 and len(grad_new.shape) == 2:
                                grad = grad.unsqueeze(0)
                            elif len(grad.shape) == 2 and len(grad_new.shape) == 1:
                                grad_new = grad_new.unsqueeze(0)
                    # Now add
                    grad = grad + grad_new.detach()  # Detach from computation graph
                # print(f'grad={grad}')
            # Calculate average entropy
            entropy = approximate_cond_entropy / trajectory_iter
            print("The approximate entropy is", entropy)
            self.entropy_list.append(entropy)

            # Calculate total cost
            total_cost = (approximate_cond_entropy / trajectory_iter + self.weight * torch.sum(self.x)).item()
            print("The objective function is", total_cost)
            self.total_cost_list.append(total_cost)

            # Average gradient
            grad = grad / trajectory_iter

            # Handle different shapes for printing - ensure detached for numpy conversion
            if len(grad.shape) == 2:
                print("The gradient of entropy", grad[0][self.modify_list].detach().cpu().numpy())
            else:
                print("The gradient of entropy", grad[self.modify_list].detach().cpu().numpy())

            # Update x in one go
            # with timing("update_x"):
            # with torch.no_grad():
            # Make sure x and grad have compatible shapes for subtraction
            if len(grad.shape) == 2 and len(self.x.shape) == 1:
                grad_to_use = grad[0]
            elif len(grad.shape) == 1 and len(self.x.shape) == 2:
                grad_to_use = grad.unsqueeze(0)
            else:
                grad_to_use = grad

            self.x = torch.clamp(self.x - eta * grad_to_use, min=0)


            # Print in the right format - ensure detached for numpy conversion
            if len(self.x.shape) == 2:
                print("The side payment is", self.x[0][self.modify_list].detach().cpu().numpy())
            else:
                print("The side payment is", self.x[self.modify_list].detach().cpu().numpy())

            # Re-initialize x - possibly redundant operation
            if len(self.x.shape) == 2:
                self.x = torch.nn.Parameter(self.x[0].detach().clone(), requires_grad=False)
            else:
                self.x = torch.nn.Parameter(self.x.detach().clone(), requires_grad=False)

            end = time.time()
            print("Time for the iteration", i, ":", end - start, "s.")
            print("#" * 100)

        # Set iteration list
        self.iteration_list = range(self.iter_num)

        # Save results
        with open(f'./Data/entropy_values_{self.ex_num}.pkl', 'wb') as file:
            pickle.dump(self.entropy_list, file)

        with open(f'./Data/x_list_{self.ex_num}', 'wb') as file:
            pickle.dump(self.x_list, file)

        with open(f'./Data/theta_collection_{self.ex_num}', 'wb') as file:
            pickle.dump(self.theta_torch_collection, file)

        plot_figures(self.ex_num,self.iter_num,self.modify_list, self.weight)

        return