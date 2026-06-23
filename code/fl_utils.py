"""
Utility components for federated learning algorithms.
This module contains helper classes and functions used by various FL algorithms.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from ddpg_page import DDPG


class SamplingLogger:
    """
    State-of-the-art sampling logger for federated learning algorithms.

    This class provides comprehensive tracking of client sampling patterns
    across communication rounds, enabling detailed analysis of client
    participation fairness and sampling distribution.

    Attributes:
        n_clients (int): Total number of clients in the federation
        sampling_counts (np.ndarray): Array tracking sampling frequency per client
        round_history (list): Optional detailed history of sampling per round

    Methods:
        record_sampling: Records client sampling for a given round
        get_sampling_counts: Returns current sampling statistics
        reset: Resets all sampling statistics
    """

    def __init__(self, n_clients: int, track_history: bool = False):
        """
        Initialize the sampling logger.

        Args:
            n_clients (int): Total number of clients in the federation
            track_history (bool): Whether to maintain detailed round-by-round history
        """
        self.n_clients = n_clients
        self.sampling_counts = np.zeros(n_clients, dtype=np.int32)
        self.track_history = track_history
        self.round_history = [] if track_history else None

    def record_sampling(
        self, sampled_client_indices: np.ndarray, round_num: int = None
    ):
        """
        Record client sampling for the current round.

        Args:
            sampled_client_indices (np.ndarray): Indices of clients sampled this round
            round_num (int, optional): Round number for history tracking
        """
        # Update sampling counts for each sampled client
        for client_idx in sampled_client_indices:
            self.sampling_counts[client_idx] += 1

        # Optionally store detailed history
        if self.track_history and round_num is not None:
            self.round_history.append(
                {"round": round_num, "sampled_clients": sampled_client_indices.copy()}
            )

    def get_sampling_counts(self) -> np.ndarray:
        """
        Get current sampling counts for all clients.

        Returns:
            np.ndarray: Array of sampling counts per client
        """
        return self.sampling_counts.copy()

    def reset(self):
        """Reset all sampling statistics."""
        self.sampling_counts.fill(0)
        if self.track_history:
            self.round_history.clear()


class ClusteredSampling:
    """
    Clustered Sampling algorithm based on Algorithm 1.

    This class implements a deterministic, weight-based client sampling strategy
    that pre-computes a static probability distribution matrix for efficient
    client selection in federated learning.
    """

    def __init__(self, all_client_weights, n_sampled, epsilon=1e10):
        """
        Initialize the Clustered Sampling instance.

        Args:
            all_client_weights (np.ndarray): A NumPy array containing normalized weights
                                            for all clients.
            n_sampled (int): Number of clients to sample per round (M).
            epsilon (float): Large number used to convert float weights to integers
                           for precise computation.
        """
        self.weights = all_client_weights
        self.n_sampled = n_sampled
        self.epsilon = epsilon
        self.n_clients = len(all_client_weights)

        # Generate the sampling distribution matrix
        self._generate_distribution()

    def _generate_distribution(self):
        """
        Private method implementing Algorithm 1's core logic to generate the
        sampling distribution matrix.

        This method uses a greedy bin-packing approach to distribute client
        weights across sampling rounds.
        """
        # Step 1: Weight amplification
        augmented_weights = self.weights * self.n_sampled * self.epsilon

        # Step 2: Sort clients by augmented weights in descending order
        sorted_indices = np.argsort(augmented_weights)[::-1]

        # Step 3: Initialize distribution matrix
        distri_clusters = np.zeros((self.n_sampled, self.n_clients), dtype=int)

        # Step 4: Greedy allocation (bin-packing)
        # Initialize box capacities (each box has capacity epsilon)
        box_used_capacity = np.zeros(self.n_sampled)

        # Current box index
        k = 0

        # Iterate through sorted clients
        for client_idx in sorted_indices:
            client_remaining_weight = augmented_weights[client_idx]

            # Allocate client weight to boxes
            while client_remaining_weight > 0 and k < self.n_sampled:
                # Calculate current box's remaining capacity
                capacity_k = self.epsilon - box_used_capacity[k]

                # Determine allocation amount
                u_i = min(capacity_k, client_remaining_weight)

                # Update distribution matrix
                # Use min to avoid overflow issues
                distri_clusters[k, client_idx] = int(min(u_i, 2**31 - 1))

                # Update client's remaining weight
                client_remaining_weight -= u_i

                # Update box's used capacity
                box_used_capacity[k] += u_i

                # If box k is full, move to next box
                if (
                    box_used_capacity[k] >= self.epsilon - 1e-6
                ):  # Small tolerance for float comparison
                    k += 1

        # Step 5: Normalize to get probability distributions
        # Convert to float and normalize each row to sum to 1
        distri_clusters = distri_clusters.astype(np.float32)
        row_sums = distri_clusters.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        self.distri_clusters = distri_clusters / row_sums

    def sample(self, seed=None):
        """
        Execute one round of sampling based on the pre-computed distribution matrix.

        Args:
            seed (int, optional): Random seed for reproducibility.

        Returns:
            list: A list containing M client indices sampled for this round.
        """
        if seed is not None:
            np.random.seed(seed)

        sampled_clients = []

        # Iterate through each row of the distribution matrix
        for k in range(self.n_sampled):
            # Sample one client from the probability distribution in row k
            client_idx = np.random.choice(
                self.n_clients, size=1, p=self.distri_clusters[k]
            )[0]
            sampled_clients.append(client_idx)

        return sampled_clients


class FedSampling:
    """
    FedSampling algorithm's privacy-preserving data size estimator.

    This class implements the differential privacy mechanism for estimating
    the total number of samples across all clients in the federated network,
    as described in the FedSampling paper.
    """

    def __init__(self, client_sample_counts: list, alpha: float, m_param: int):
        """
        Initialize the FedSampling estimator.

        Args:
            client_sample_counts: List containing the number of training samples for each client
            alpha: Privacy parameter controlling the probability of true vs random response (0.0-1.0)
            m_param: Upper bound for random response values
        """
        self.client_sample_counts = client_sample_counts
        self.num_clients = len(client_sample_counts)
        self.alpha = alpha
        self.m_param = m_param

        # Precompute total real samples for robustness check
        self.total_real_samples = sum(client_sample_counts)

        # Validate parameters
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Alpha must be in [0.0, 1.0], got {alpha}")
        if m_param <= 1:
            raise ValueError(f"m_param must be > 1, got {m_param}")
        if self.num_clients == 0:
            raise ValueError("client_sample_counts cannot be empty")

    def query(self, client_id: int) -> int:
        """
        Perform a privacy-preserving query on a single client's data size.

        This implements the randomized response mechanism where:
        - With probability alpha: return the true sample count
        - With probability (1-alpha): return a random value in [1, m_param)

        Args:
            client_id: Index of the client to query

        Returns:
            int: Either the true sample count or a random response
        """
        if not 0 <= client_id < self.num_clients:
            raise ValueError(
                f"Invalid client_id {client_id}, must be in [0, {self.num_clients})"
            )

        # Generate random response in range [1, m_param)
        fake_response = np.random.randint(1, self.m_param)

        # Get true response
        real_response = self.client_sample_counts[client_id]

        # Apply differential privacy mechanism
        if np.random.random() < self.alpha:
            return real_response
        else:
            return fake_response

    def estimate(self) -> float:
        """
        Estimate the total number of samples across all clients.

        This method queries all clients and applies bias correction to obtain
        an unbiased estimate of the total data size in the federated network.

        Returns:
            float: Estimated total number of samples (hat_N)
        """
        # Collect responses from all clients
        R = 0
        for client_id in range(self.num_clients):
            R += self.query(client_id)

        # Apply bias correction formula
        # E[fake_response] = m_param / 2 (expected value of uniform distribution on [1, m_param))
        expected_fake = self.m_param / 2.0

        # Bias correction: hat_N = (R - n*(1-alpha)*E[fake]) / alpha
        hat_N = (R - self.num_clients * (1 - self.alpha) * expected_fake) / self.alpha

        # Ensure robustness: the estimate should be at least as large as the true total
        # This prevents negative or unreasonably small estimates due to randomness
        return max(hat_N, self.total_real_samples)


class PAGE:
    """
    PAGE (Personalized and Generalized balancing Game) Manager.

    This class encapsulates all Deep Reinforcement Learning (DRL) logic for the PAGE algorithm,
    managing both client and server DDPG agents to solve the two-tier cooperative-competitive game
    in federated learning.
    """

    def __init__(self, args, W):
        """
        Initialize PAGE_Manager with client and server DDPG agents.

        Args:
            args: Command line arguments containing all hyperparameters
            W: Total number of clients in the system
        """
        self.args = args
        self.W = W

        # Define RL environment dimensions
        self.c_nb_states = 1  # Client state: local model test accuracy
        self.c_nb_actions = 2  # Client actions: [learning_rate, epochs]
        self.s_nb_states = W  # Server state: all clients' test accuracies
        self.s_nb_actions = W  # Server actions: aggregation weights for all clients

        # Initialize client DDPG agent
        # Client agent manages all W clients collectively
        self.client_agent = DDPG(
            nb_states=self.c_nb_states,
            nb_actions=self.c_nb_actions,
            nb_agents=W,
            args=args,
            hidden1=args.page_c_hidden1,
            hidden2=args.page_c_hidden2,
            actor_lr=args.page_c_actor_lr,
            critic_lr=args.page_c_critic_lr,
            device="cuda" if args.device == "gpu" else "cpu",
        )

        # Initialize server DDPG agent
        # Server is a single agent that outputs weights for all clients
        self.server_agent = DDPG(
            nb_states=self.s_nb_states,
            nb_actions=self.s_nb_actions,
            nb_agents=1,  # Server is a single agent
            args=args,
            hidden1=args.page_s_hidden1,
            hidden2=args.page_s_hidden2,
            actor_lr=args.page_s_actor_lr,
            critic_lr=args.page_s_critic_lr,
            device="cuda" if args.device == "gpu" else "cpu",
        )

        # Initialize state and performance tracking variables
        self.client_observation = np.zeros((W, self.c_nb_states))
        self.server_observation = np.zeros((1, self.s_nb_states))
        self.last_avg_pers_acc = 0.0
        self.last_global_acc = 0.0

    def get_actions(self, current_round, warmup_rounds):
        """
        Generate actions for clients and server based on current round.

        Args:
            current_round: Current federated learning communication round
            warmup_rounds: Number of warmup rounds for RL

        Returns:
            tuple: (client_actions, server_weights)
                - client_actions: np.ndarray of shape (W, 2) with [lr, epochs] for each client
                - server_weights: np.ndarray of shape (W,) with aggregation weights
        """
        if current_round <= warmup_rounds:
            # Warmup phase: use random actions for exploration
            client_actions = self.client_agent.random_a()
            server_weights = self.server_agent.random_sa()
        else:
            # Learning phase: use learned policy
            client_actions = self.client_agent.select_a(self.client_observation)
            server_weights = self.server_agent.select_sa(self.server_observation)

        # Ensure proper formatting
        client_actions = np.squeeze(client_actions)
        if client_actions.ndim == 1:
            client_actions = client_actions.reshape(1, -1)

        server_weights = np.squeeze(server_weights)

        # Normalize server weights to ensure they sum to 1
        server_weights = np.abs(server_weights)
        server_weights = server_weights / (np.sum(server_weights) + 1e-8)

        return client_actions, server_weights

    def update_and_learn(
        self,
        new_client_accuracies,
        new_global_accuracy,
        current_round,
        warmup_rounds,
        done_flag,
    ):
        """
        Update states, compute rewards, and trigger DDPG learning.

        Args:
            new_client_accuracies: np.ndarray of shape (W,) with personalized accuracies
            new_global_accuracy: float, global model test accuracy
            current_round: Current round number
            warmup_rounds: Number of warmup rounds
            done_flag: Boolean indicating if FL process is complete
        """
        # Step 1: Compute rewards
        # Client rewards: improvement in individual accuracy
        client_rewards = new_client_accuracies - self.client_observation.flatten()

        # Server reward: weighted combination of global and personalized improvements
        avg_pers_acc = np.mean(new_client_accuracies)
        pers_reward = avg_pers_acc - self.last_avg_pers_acc
        global_reward = new_global_accuracy - self.last_global_acc

        # Combine rewards using alpha parameter
        server_reward = (
            self.args.page_reward_alpha * global_reward
            + (1 - self.args.page_reward_alpha) * pers_reward
        )

        # Step 2: Construct new observations
        new_client_observation = new_client_accuracies.reshape(-1, 1)
        new_server_observation = new_client_accuracies.reshape(1, -1)

        # Step 3: Let agents observe the transition
        self.client_agent.observe(client_rewards, new_client_observation, done_flag)
        self.server_agent.observe([server_reward], new_server_observation, done_flag)

        # Step 4: Update policies if past warmup phase
        if current_round > warmup_rounds:
            self.client_agent.update_policy()
            self.server_agent.update_policy()

        # Step 5: Update internal state
        self.client_observation = new_client_observation
        self.server_observation = new_server_observation
        self.last_avg_pers_acc = avg_pers_acc
        self.last_global_acc = new_global_accuracy


class FedALA:
    """
    Adaptive Local Aggregation (ALA) module for FedALA algorithm.

    This class implements the adaptive weight learning mechanism that allows
    each client to learn personalized aggregation weights for combining
    global and local models based on their local data distribution.
    """

    def __init__(
        self,
        cid,
        loss,
        train_data,
        batch_size,
        rand_percent,
        layer_idx,
        eta,
        device,
        threshold,
        num_pre_loss,
    ):
        """
        Initialize ALA module for a specific client.

        Args:
            cid: Client ID
            loss: Loss function (e.g., nn.CrossEntropyLoss())
            train_data: Client's local training dataset
            batch_size: Batch size for weight learning
            rand_percent: Percentage of local data to use for weight learning
            layer_idx: Layer index for adaptive aggregation (0 for all layers)
            eta: Learning rate for weight optimization
            device: Device to run computations on (CPU/GPU)
            threshold: Convergence threshold for loss variance
            num_pre_loss: Number of recent losses to track for convergence
        """
        self.cid = cid
        self.loss = loss
        self.train_data = train_data
        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.device = device
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss

        # Initialize tracking for computation cost
        self.last_run_epochs = 0

        # Initialize aggregation weights (will be set based on model structure)
        self.weights = None
        self.start_phase = True

    def adaptive_local_aggregation(self, global_model, local_model):
        """
        Perform adaptive local aggregation to create a personalized model.

        This method learns optimal weights for combining global and local models
        based on the client's local data distribution.

        Args:
            global_model: The global model from the server
            local_model: The client's local model (will be modified in-place)
        """
        # Get model parameters
        global_params = list(global_model.parameters())
        local_params = list(local_model.parameters())

        # Initialize weights if first time
        if self.weights is None or self.start_phase:
            self.weights = self._initialize_weights(global_params)
            self.start_phase = False

        # Create a temporary model for weight learning
        temp_model = copy.deepcopy(local_model)
        temp_params = list(temp_model.parameters())

        # Sample a subset of training data for weight learning
        num_samples = int(len(self.train_data) * self.rand_percent / 100)
        if num_samples == 0:
            num_samples = min(len(self.train_data), self.batch_size)

        # Random sampling of data indices
        sample_indices = np.random.choice(
            len(self.train_data), num_samples, replace=False
        )

        # Create data loader for sampled data
        if hasattr(self.train_data, "dataset"):
            # If train_data is a DataLoader
            sampled_data = torch.utils.data.Subset(
                self.train_data.dataset, sample_indices
            )
        else:
            # If train_data is a Dataset
            sampled_data = torch.utils.data.Subset(self.train_data, sample_indices)

        data_loader = torch.utils.data.DataLoader(
            sampled_data, batch_size=min(self.batch_size, num_samples), shuffle=True
        )

        # Optimization loop for learning aggregation weights
        optimizer = optim.SGD(self.weights.values(), lr=self.eta)

        # Track losses for convergence check
        losses = []
        cnt = 0

        while True:
            epoch_loss = 0.0
            num_batches = 0

            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Aggregate parameters using current weights
                self._aggregate_parameters(temp_params, global_params, local_params)

                # Forward pass
                temp_model.zero_grad()
                output = temp_model(batch_x)
                loss_val = self.loss(output, batch_y)

                # Backward pass for weight optimization
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

                # Ensure weights remain in valid range [0, 1]
                with torch.no_grad():
                    for key in self.weights:
                        self.weights[key].clamp_(0.0, 1.0)

                epoch_loss += loss_val.item()
                num_batches += 1

            # Average loss for this epoch
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            losses.append(avg_loss)
            cnt += 1

            # Check convergence
            if len(losses) >= self.num_pre_loss:
                recent_losses = losses[-self.num_pre_loss :]
                loss_std = np.std(recent_losses)
                if loss_std < self.threshold:
                    break

            # Maximum iterations safeguard
            if cnt >= 100:  # Reasonable upper bound
                break

        # Store the number of epochs for computation tracking
        self.last_run_epochs = cnt

        # Apply the learned weights to create the final personalized model
        with torch.no_grad():
            self._aggregate_parameters(local_params, global_params, local_params)

    def _initialize_weights(self, params):
        """
        Initialize aggregation weights based on model structure.

        Args:
            params: List of model parameters

        Returns:
            dict: Dictionary of learnable weight parameters
        """
        weights = {}

        if self.layer_idx == 0:
            # Use weights for all layers
            for i, param in enumerate(params):
                if param.requires_grad:
                    # Initialize weight to 0.5 (equal contribution from global and local)
                    weights[f"layer_{i}"] = nn.Parameter(
                        torch.tensor(0.5, device=self.device, requires_grad=True)
                    )
        else:
            # Use weight only for specific layer (counting from the end)
            target_idx = len(params) - self.layer_idx
            if 0 <= target_idx < len(params) and params[target_idx].requires_grad:
                weights[f"layer_{target_idx}"] = nn.Parameter(
                    torch.tensor(0.5, device=self.device, requires_grad=True)
                )

        return weights

    def _aggregate_parameters(self, target_params, global_params, local_params):
        """
        Aggregate global and local parameters using learned weights.

        Args:
            target_params: Parameters to update (modified in-place)
            global_params: Global model parameters
            local_params: Local model parameters
        """
        with torch.no_grad():
            for i, (target, global_p, local_p) in enumerate(
                zip(target_params, global_params, local_params)
            ):
                # Check if we have a weight for this layer
                weight_key = f"layer_{i}"
                if weight_key in self.weights:
                    # Use learned weight for aggregation
                    w = self.weights[weight_key]
                    target.data = w * local_p.data + (1 - w) * global_p.data
                else:
                    # If no weight for this layer, use global parameters
                    target.data = global_p.data.clone()
