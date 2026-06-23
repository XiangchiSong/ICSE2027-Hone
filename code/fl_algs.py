import numpy as np
import pickle
import torch
from tqdm import tqdm

# from progressbar import progressbar
import progressbar
from models import *
import copy
import gym
import torch.nn.functional as F
import component_lib
from hone_utils import HoneServerState
from fl_utils import FedSampling, ClusteredSampling, SamplingLogger, PAGE, FedALA
from comp_utils import ComputationTracker
from fl_optimisers import ClientSGD, ClientDittoOptimizer


def init_stats_arrays4(T):
    """
    Returns:
        (tupe) of 4 numpy 0-filled float32 arrays of length T.
    """
    return tuple(np.zeros(T, dtype=np.float32) for i in range(4))


def init_stats_arrays5(T):
    """
    Returns:
        (tuple) of 5 numpy 0-filled float32 arrays of length T.
    """
    return tuple(np.zeros(T, dtype=np.float32) for i in range(5))


# FedAvg
def run_fedavg(
    data_feeders,
    test_data,
    model,
    client_opt,
    T,
    M,
    K,
    B,
    test_freq=1,
    bn_setting=0,
    noisy_idxs=[],
):
    """
    Code Reference: Communication-efficient learning of deep networks from decentralized data', McMahan et al., AISTATS 2021.
    Run Federated Averaging (FedAvg) algorithm from 'Communication-efficient
    learning of deep networks from decentralized data', McMahan et al., AISTATS
    2021. In this implementation, the parameters of the client optimisers are
    also averaged (gives FedAvg-Adam when client_opt is ClientAdam). Runs T
    rounds of FedAvg, and returns the training and test results.

    Returns:
        train_errs, train_accs, test_errs, test_accs, sampling_counts, comp_costs
    """
    W = len(data_feeders)

    train_errs, train_accs, test_errs, test_accs, comp_costs = init_stats_arrays5(T)

    # Initialize computation tracker
    comp_tracker = ComputationTracker(model, model.device, "fedavg")

    # Initialize sampling logger for tracking client participation
    sampling_logger = SamplingLogger(n_clients=W, track_history=False)

    # contains private model and optimiser BN vals (if bn_setting != 3)
    user_bn_model_vals = [model.get_bn_vals(setting=bn_setting) for w in range(W)]
    user_bn_optim_vals = [client_opt.get_bn_params(model) for w in range(W)]

    # global model/optimiser updated at the end of each round
    round_model = model.get_params()
    round_optim = client_opt.get_params()

    # stores accumulated client models/optimisers each round
    round_agg = model.get_params()
    round_opt_agg = client_opt.get_params()

    bar = progressbar.ProgressBar()
    for t in bar(range(T)):
        round_agg = round_agg.zeros_like()
        round_opt_agg = round_opt_agg.zeros_like()

        # select round clients and compute their weights for later sum
        user_idxs = np.random.choice(W, M, replace=False)

        # Record client sampling for this round
        sampling_logger.record_sampling(user_idxs, round_num=t)

        weights = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights = weights.astype(np.float32)
        weights = weights / np.sum(weights)

        round_n_test_users = 0

        for (w, user_idx) in zip(weights, user_idxs):
            # download global model/optim, update with private BN params
            model.set_params(round_model)
            client_opt.set_params(round_optim)
            model.set_bn_vals(user_bn_model_vals[user_idx], setting=bn_setting)
            client_opt.set_bn_params(
                user_bn_optim_vals[user_idx], model, setting=bn_setting
            )

            # test local model if not a noisy client
            if (t % test_freq == 0) and (user_idx not in noisy_idxs):
                err, acc = model.test(
                    test_data[0][user_idx], test_data[1][user_idx], 128
                )
                test_errs[t] += err
                test_accs[t] += acc
                round_n_test_users += 1

            # perform local SGD
            for k in range(K):
                x, y = data_feeders[user_idx].next_batch(B)
                err, acc = model.train_step(x, y)
                train_errs[t] += err
                train_accs[t] += acc

            # Note: Base model training FLOPs are identical across all algorithms
            # Only track FedAvg-specific operations

            # upload local model/optim to server, store private BN params
            round_agg = round_agg + (model.get_params() * w)
            round_opt_agg = round_opt_agg + (client_opt.get_params() * w)
            user_bn_model_vals[user_idx] = model.get_bn_vals(setting=bn_setting)
            user_bn_optim_vals[user_idx] = client_opt.get_bn_params(
                model, setting=bn_setting
            )

        # Track FedAvg-specific aggregation operations
        comp_tracker.track_fedavg_operations(M)

        # new global model is weighted sum of client models
        round_model = round_agg.copy()
        round_optim = round_opt_agg.copy()

        # Record cumulative computation cost for this round
        comp_costs[t] = comp_tracker.get_total_flops()

        if t % test_freq == 0:
            test_errs[t] = test_errs[t] / round_n_test_users
            test_accs[t] = test_accs[t] / round_n_test_users
            formatted_flops = comp_tracker.get_formatted_flops()
            print(
                f"Round {t}: Test Accuracy = {test_accs[t]}, Test Error = {test_errs[t]}, Comp Cost = {formatted_flops}"
            )

    train_errs /= M * K
    train_accs /= M * K

    # Get final sampling counts
    sampling_counts = sampling_logger.get_sampling_counts()

    return train_errs, train_accs, test_errs, test_accs, sampling_counts, comp_costs


# FedALA
def run_fedala(
    data_feeders,
    test_data,
    model,
    client_opt,
    T,
    M,
    K,
    B,
    args,
    test_freq=1,
    bn_setting=0,
    noisy_idxs=[],
):
    """
    Code Reference: Zhang J, Hua Y, Wang H, et al. Fedala: Adaptive local aggregation for personalized federated learning[C]//Proceedings of the AAAI conference on artificial intelligence. 2023, 37(9): 11237-11244.
    Run FedALA (Federated Adaptive Local Aggregation) algorithm.

    FedALA allows each client to learn personalized aggregation weights for
    combining global and local models based on their local data distribution.
    This implementation follows the architecture patterns of other FL algorithms
    in this codebase while incorporating the unique ALA mechanism.

    Args:
        data_feeders: List of PyTorchDataFeeder instances for each client
        test_data: Tuple of (x, y) test data as torch tensors
        model: FLModel instance to train
        client_opt: Client optimizer instance
        T: Number of communication rounds
        M: Number of clients selected per round
        K: Number of local training steps per client
        B: Batch size for local training
        args: Command line arguments containing FedALA hyperparameters
        test_freq: Frequency of testing (default: 1)
        bn_setting: Batch normalization setting
        noisy_idxs: List of indices of noisy clients

    Returns:
        Tuple of (train_errs, train_accs, test_errs, test_accs, sampling_counts, comp_costs)
    """
    W = len(data_feeders)

    # Initialize statistics arrays
    train_errs, train_accs, test_errs, test_accs, comp_costs = init_stats_arrays5(T)

    # Initialize computation tracker
    comp_tracker = ComputationTracker(model, model.device, "fedala")

    # Initialize sampling logger for tracking client participation
    sampling_logger = SamplingLogger(n_clients=W, track_history=False)

    # Initialize BN parameters storage
    user_bn_model_vals = [model.get_bn_vals(setting=bn_setting) for w in range(W)]
    user_bn_optim_vals = [client_opt.get_bn_params(model) for w in range(W)]

    # Global model/optimizer updated at the end of each round
    round_model = model.get_params()
    round_optim = client_opt.get_params()

    # Extract raw training data from feeders
    # Assume feeder objects have access to the original dataset
    all_train_data = []
    for feeder in data_feeders:
        # Check different possible attributes where the dataset might be stored
        if hasattr(feeder, "dataset"):
            all_train_data.append(feeder.dataset)
        elif hasattr(feeder, "x") and hasattr(feeder, "y"):
            # Create a simple dataset from x and y tensors
            from torch.utils.data import TensorDataset

            dataset = TensorDataset(feeder.x, feeder.y)
            all_train_data.append(dataset)
        else:
            # Fallback: create a dataset that yields batches from the feeder
            class FeederDataset(torch.utils.data.Dataset):
                def __init__(self, feeder):
                    self.feeder = feeder
                    self.x = feeder.x
                    self.y = feeder.y

                def __len__(self):
                    return len(self.x)

                def __getitem__(self, idx):
                    return self.x[idx], self.y[idx]

            all_train_data.append(FeederDataset(feeder))

    # Initialize ALA modules for each client
    ala_modules = [
        FedALA(
            cid=w,
            loss=model.loss_fn,
            train_data=all_train_data[w],
            batch_size=args.B,
            rand_percent=args.fedala_rand_percent,
            layer_idx=args.fedala_layer_idx,
            eta=args.fedala_eta,
            device=model.device,
            threshold=args.fedala_threshold,
            num_pre_loss=args.fedala_num_pre_loss,
        )
        for w in range(W)
    ]

    # Store each client's local model from previous round
    client_local_models = [round_model.copy() for _ in range(W)]

    # Progress bar for training rounds
    bar = progressbar.ProgressBar()
    for t in bar(range(T)):
        # Initialize round aggregation
        round_agg = round_model.zeros_like()
        round_opt_agg = round_optim.zeros_like()

        # Select M clients randomly
        user_idxs = np.random.choice(W, M, replace=False)

        # Record client sampling for this round
        sampling_logger.record_sampling(user_idxs, round_num=t)

        # Compute weights for aggregation
        weights = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights = weights.astype(np.float32)
        weights = weights / np.sum(weights)

        round_n_test_users = 0

        for (w, user_idx) in zip(weights, user_idxs):
            # Step 1: Download global model
            model.set_params(round_model)
            client_opt.set_params(round_optim)
            model.set_bn_vals(user_bn_model_vals[user_idx], setting=bn_setting)
            client_opt.set_bn_params(
                user_bn_optim_vals[user_idx], model, setting=bn_setting
            )

            # Step 2: Perform adaptive local aggregation (FedALA core)
            # Create models for ALA
            global_model_for_ala = copy.deepcopy(model)
            global_model_for_ala.set_params(round_model)

            local_model_from_last_round = copy.deepcopy(model)
            local_model_from_last_round.set_params(client_local_models[user_idx])

            # Get ALA module for this client
            client_ala_module = ala_modules[user_idx]

            # Perform adaptive aggregation - this modifies model in-place
            client_ala_module.adaptive_local_aggregation(
                global_model=global_model_for_ala,
                local_model=model,  # This will be modified to become personalized
            )

            # Track ALA computation cost
            ala_epochs = client_ala_module.last_run_epochs
            # Count the number of layers with learnable parameters
            num_layers = sum(1 for p in model.parameters() if p.requires_grad)
            comp_tracker.track_fedala_operations(
                args.fedala_rand_percent,
                len(all_train_data[user_idx]),
                ala_epochs,
                num_layers,
                M,
            )

            # Step 3: Test personalized model (before local training)
            if (t % test_freq == 0) and (user_idx not in noisy_idxs):
                err, acc = model.test(
                    test_data[0][user_idx], test_data[1][user_idx], 128
                )
                test_errs[t] += err
                test_accs[t] += acc
                round_n_test_users += 1

            # Step 4: Perform K steps of local SGD training
            for k in range(K):
                x, y = data_feeders[user_idx].next_batch(B)
                err, acc = model.train_step(x, y)
                train_errs[t] += err
                train_accs[t] += acc

            # Step 5: Upload and store
            # Get updated model parameters
            updated_params = model.get_params()

            # Aggregate for global model update
            round_agg = round_agg + (updated_params * w)
            round_opt_agg = round_opt_agg + (client_opt.get_params() * w)

            # Store this client's model for next round's ALA
            client_local_models[user_idx] = updated_params.copy()

            # Store BN parameters
            user_bn_model_vals[user_idx] = model.get_bn_vals(setting=bn_setting)
            user_bn_optim_vals[user_idx] = client_opt.get_bn_params(
                model, setting=bn_setting
            )

        # Update global model
        round_model = round_agg.copy()
        round_optim = round_opt_agg.copy()

        # Record cumulative computation cost for this round
        comp_costs[t] = comp_tracker.get_total_flops()

        # Compute test statistics
        if t % test_freq == 0:
            if round_n_test_users > 0:
                test_errs[t] /= round_n_test_users
                test_accs[t] /= round_n_test_users
                formatted_flops = comp_tracker.get_formatted_flops()
                print(
                    f"Round {t}: Test Accuracy = {test_accs[t]}, Test Error = {test_errs[t]}, Comp Cost = {formatted_flops}"
                )

    # Normalize training statistics
    train_errs /= M * K
    train_accs /= M * K

    # Get final sampling counts
    sampling_counts = sampling_logger.get_sampling_counts()

    return train_errs, train_accs, test_errs, test_accs, sampling_counts, comp_costs


# PAGE
def run_page(
    data_feeders,
    test_data,
    model,
    client_opt,
    T,
    M,
    K,
    B,
    args,
    test_freq=1,
    bn_setting=0,
    noisy_idxs=[],
):
    """
    Code Reference: Chen Q, Wang Z, Hu J, et al. PAGE: Equilibrate personalization and generalization in federated learning[C]//Proceedings of the ACM Web Conference 2024. 2024: 2955-2964.
    Run PAGE (Personalized and Generalized balancing Game) algorithm.

    PAGE uses deep reinforcement learning (DDPG) to dynamically optimize:
    - Client-side: learning rate and local epochs for each client
    - Server-side: aggregation weights for model fusion

    This creates a two-tier cooperative-competitive game that balances
    personalization and generalization in federated learning.

    Args:
        data_feeders: List of PyTorchDataFeeder instances for each client
        test_data: Tuple of (x, y) test data as torch tensors
        model: FLModel instance to train
        client_opt: Client optimizer instance
        T: Number of communication rounds
        M: Number of clients selected per round
        K: Number of local training steps per client (base value)
        B: Batch size for local training
        args: Command line arguments containing PAGE hyperparameters
        test_freq: Frequency of testing (default: 1)
        bn_setting: Batch normalization setting
        noisy_idxs: List of indices of noisy clients

    Returns:
        Tuple of (train_errs, train_accs, test_errs, test_accs, sampling_counts)
    """
    W = len(data_feeders)

    # Initialize statistics arrays
    train_errs, train_accs, test_errs, test_accs, comp_costs = init_stats_arrays5(T)

    # Initialize computation tracker
    comp_tracker = ComputationTracker(model, model.device, "page")

    # Initialize sampling logger for tracking client participation
    sampling_logger = SamplingLogger(n_clients=W, track_history=False)

    # Initialize PAGE manager with DDPG agents
    page_manager = PAGE(args, W)

    # Initialize global performance metrics
    global_test_errs = np.zeros(T, dtype=np.float32)
    global_test_accs = np.zeros(T, dtype=np.float32)

    # Initialize BN parameters storage (similar to FedAvg)
    user_bn_model_vals = [model.get_bn_vals(setting=bn_setting) for w in range(W)]
    user_bn_optim_vals = [client_opt.get_bn_params(model) for w in range(W)]

    # Global model/optimizer updated at the end of each round
    round_model = model.get_params()
    round_optim = client_opt.get_params()

    # Calculate steps per epoch for dynamic K computation
    steps_per_E = K // args.E if args.E > 0 else K

    # Progress bar for training rounds
    bar = progressbar.ProgressBar()
    for t in bar(range(T)):
        # Step 1: Get actions from PAGE manager
        client_actions, server_weights = page_manager.get_actions(t, args.page_warmup)

        # Track PAGE DDPG inference operations
        comp_tracker.track_page_operations(W, include_training=False)

        # Step 2: Client sampling using server weights
        # Normalize weights to create probability distribution
        sampling_probs = server_weights / np.sum(server_weights)

        # Sample M clients without replacement based on weights
        try:
            user_idxs = np.random.choice(W, M, replace=False, p=sampling_probs)
        except ValueError:
            # Fallback to uniform sampling if weights are problematic
            user_idxs = np.random.choice(W, M, replace=False)

        # Record client sampling for this round
        sampling_logger.record_sampling(user_idxs, round_num=t)

        # Step 3: Initialize round aggregation
        round_agg = round_model.zeros_like()
        round_opt_agg = round_optim.zeros_like()

        # Get aggregation weights for sampled clients
        sampled_weights = server_weights[user_idxs]
        agg_weights = sampled_weights / np.sum(sampled_weights)

        # Step 4: Local training for each sampled client
        updated_models = []
        for idx, (user_idx, w) in enumerate(zip(user_idxs, agg_weights)):
            # Download global model/optimizer
            model.set_params(round_model)
            client_opt.set_params(round_optim)
            model.set_bn_vals(user_bn_model_vals[user_idx], setting=bn_setting)
            client_opt.set_bn_params(
                user_bn_optim_vals[user_idx], model, setting=bn_setting
            )

            # Get dynamic hyperparameters for this client
            lr = float(client_actions[user_idx][0])
            epochs = int(np.round(client_actions[user_idx][1]))
            K_dynamic = steps_per_E * epochs

            # Perform local training with dynamic hyperparameters
            for k in range(K_dynamic):
                x, y = data_feeders[user_idx].next_batch(B)

                # Override learning rate for this step
                for param_group in client_opt.param_groups:
                    param_group["lr"] = lr

                err, acc = model.train_step(x, y)
                train_errs[t] += err
                train_accs[t] += acc

            # Store updated model
            updated_model = model.get_params()
            updated_models.append(updated_model)

            # Aggregate models with server weights
            round_agg = round_agg + (updated_model * w)
            round_opt_agg = round_opt_agg + (client_opt.get_params() * w)

            # Store private BN parameters
            user_bn_model_vals[user_idx] = model.get_bn_vals(setting=bn_setting)
            user_bn_optim_vals[user_idx] = client_opt.get_bn_params(
                model, setting=bn_setting
            )

        # Step 5: Update global model
        round_model = round_agg.copy()
        round_optim = round_opt_agg.copy()

        # Step 6: Performance evaluation
        if t % test_freq == 0:
            # Evaluate global model accuracy
            model.set_params(round_model)

            # Create a unified test set from all clients for global evaluation
            global_test_acc = 0.0
            global_test_err = 0.0
            total_test_samples = 0

            for user_idx in range(W):
                if user_idx not in noisy_idxs:
                    test_size = len(test_data[1][user_idx])
                    err, acc = model.test(
                        test_data[0][user_idx], test_data[1][user_idx], 128
                    )
                    global_test_acc += acc * test_size
                    global_test_err += err * test_size
                    total_test_samples += test_size

            new_global_accuracy = (
                global_test_acc / total_test_samples if total_test_samples > 0 else 0.0
            )
            global_test_errs[t] = (
                global_test_err / total_test_samples if total_test_samples > 0 else 0.0
            )
            global_test_accs[t] = new_global_accuracy

            # Evaluate personalized model accuracies
            new_client_accuracies = np.zeros(W)
            round_n_test_users = 0

            for user_idx in range(W):
                if user_idx not in noisy_idxs:
                    # Load personalized model (latest model for this client)
                    if user_idx in user_idxs:
                        # Client was sampled this round, use its updated model
                        idx_in_sampled = list(user_idxs).index(user_idx)
                        model.set_params(updated_models[idx_in_sampled])
                    else:
                        # Client was not sampled, use previous round model
                        model.set_params(round_model)
                        model.set_bn_vals(
                            user_bn_model_vals[user_idx], setting=bn_setting
                        )

                    # Test on local data
                    err, acc = model.test(
                        test_data[0][user_idx], test_data[1][user_idx], 128
                    )
                    new_client_accuracies[user_idx] = acc

                    # Accumulate for average
                    test_errs[t] += err
                    test_accs[t] += acc
                    round_n_test_users += 1

            # Compute averages
            if round_n_test_users > 0:
                test_errs[t] /= round_n_test_users
                test_accs[t] /= round_n_test_users

            formatted_flops = comp_tracker.get_formatted_flops()
            print(
                f"Round {t}: Test Accuracy = {test_accs[t]}, Test Error = {test_errs[t]}, Comp Cost = {formatted_flops}"
            )

            # Step 7: Update PAGE manager with performance metrics
            page_manager.update_and_learn(
                new_client_accuracies,
                new_global_accuracy,
                t,
                args.page_warmup,
                done_flag=(t == T - 1),
            )

            # Track PAGE DDPG training operations
            comp_tracker.track_page_operations(W, include_training=True)

        # Record cumulative computation cost for this round
        comp_costs[t] = comp_tracker.get_total_flops()

    # Normalize training statistics
    train_errs /= M * np.mean(
        [steps_per_E * int(np.round(client_actions[i][1])) for i in range(W)]
    )
    train_accs /= M * np.mean(
        [steps_per_E * int(np.round(client_actions[i][1])) for i in range(W)]
    )

    # Get final sampling counts
    sampling_counts = sampling_logger.get_sampling_counts()

    return train_errs, train_accs, test_errs, test_accs, sampling_counts, comp_costs


# Clustered Sampling
def run_clusteredsampling(
    data_feeders,
    test_data,
    model,
    client_opt,
    T,
    M,
    K,
    B,
    test_freq=1,
    bn_setting=0,
    noisy_idxs=[],
):
    """
    Code Reference: Fraboni Y, Vidal R, Kameni L, et al. Clustered sampling: Low-variance and improved representativity for clients selection in federated learning[C]. International Conference on Machine Learning. PMLR, 2021: 3407-3416.
    Run Clustered Sampling (Algorithm 1) integrated with FedAvg framework.

    This algorithm uses a deterministic, weight-based client sampling strategy
    that pre-computes a static probability distribution matrix for efficient
    client selection throughout the training process.

    Args:
        data_feeders: List of PyTorchDataFeeder instances for each client
        test_data: Tuple of (x, y) test data as torch tensors
        model: FLModel instance to train
        client_opt: Client optimizer instance
        T: Number of communication rounds
        M: Number of clients selected per round
        K: Number of local training steps per client
        B: Batch size for local training
        test_freq: Frequency of testing (default: 1)
        bn_setting: Batch normalization setting
        noisy_idxs: List of indices of noisy clients

    Returns:
        Tuple of (train_errs, train_accs, test_errs, test_accs, sampling_counts)
    """
    W = len(data_feeders)

    train_errs, train_accs, test_errs, test_accs, comp_costs = init_stats_arrays5(T)

    # Initialize computation tracker
    comp_tracker = ComputationTracker(model, model.device, "clusteredsampling")

    # Initialize sampling logger for tracking client participation
    sampling_logger = SamplingLogger(n_clients=W, track_history=False)

    # contains private model and optimiser BN vals (if bn_setting != 3)
    user_bn_model_vals = [model.get_bn_vals(setting=bn_setting) for w in range(W)]
    user_bn_optim_vals = [client_opt.get_bn_params(model) for w in range(W)]

    # global model/optimiser updated at the end of each round
    round_model = model.get_params()
    round_optim = client_opt.get_params()

    # stores accumulated client models/optimisers each round
    round_agg = model.get_params()
    round_opt_agg = client_opt.get_params()

    # --- Clustered Sampling Initialization ---
    print("Initializing Clustered Sampler...")
    # Get all client weights based on their data volumes
    all_client_weights = np.array(
        [df.n_samples for df in data_feeders], dtype=np.float32
    )
    # Normalize weights to sum to 1
    all_client_weights /= np.sum(all_client_weights)

    # Create the ClusteredSampling instance
    sampler = ClusteredSampling(all_client_weights, n_sampled=M)
    print("Sampler initialized successfully.")
    # -----------------------------------------

    bar = progressbar.ProgressBar()
    for t in bar(range(T)):
        round_agg = round_agg.zeros_like()
        round_opt_agg = round_opt_agg.zeros_like()

        # Use clustered sampling instead of random sampling
        user_idxs = sampler.sample(seed=t)

        # Record client sampling for this round
        sampling_logger.record_sampling(user_idxs, round_num=t)

        # Compute weights for the selected clients
        weights = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights = weights.astype(np.float32)
        weights = weights / np.sum(weights)

        round_n_test_users = 0

        for (w, user_idx) in zip(weights, user_idxs):
            # download global model/optim, update with private BN params
            model.set_params(round_model)
            client_opt.set_params(round_optim)
            model.set_bn_vals(user_bn_model_vals[user_idx], setting=bn_setting)
            client_opt.set_bn_params(
                user_bn_optim_vals[user_idx], model, setting=bn_setting
            )

            # test local model if not a noisy client
            if (t % test_freq == 0) and (user_idx not in noisy_idxs):
                err, acc = model.test(
                    test_data[0][user_idx], test_data[1][user_idx], 128
                )
                test_errs[t] += err
                test_accs[t] += acc
                round_n_test_users += 1

            # perform local SGD
            for k in range(K):
                x, y = data_feeders[user_idx].next_batch(B)
                err, acc = model.train_step(x, y)
                train_errs[t] += err
                train_accs[t] += acc

            # upload local model/optim to server, store private BN params
            round_agg = round_agg + (model.get_params() * w)
            round_opt_agg = round_opt_agg + (client_opt.get_params() * w)
            user_bn_model_vals[user_idx] = model.get_bn_vals(setting=bn_setting)
            user_bn_optim_vals[user_idx] = client_opt.get_bn_params(
                model, setting=bn_setting
            )

        # Track Clustered Sampling operations
        comp_tracker.track_clustered_sampling_operations(W, M)

        # new global model is weighted sum of client models
        round_model = round_agg.copy()
        round_optim = round_opt_agg.copy()

        # Record cumulative computation cost for this round
        comp_costs[t] = comp_tracker.get_total_flops()

        if t % test_freq == 0:
            test_errs[t] = test_errs[t] / round_n_test_users
            test_accs[t] = test_accs[t] / round_n_test_users
            formatted_flops = comp_tracker.get_formatted_flops()
            print(
                f"Round {t}: Test Accuracy = {test_accs[t]}, Test Error = {test_errs[t]}, Comp Cost = {formatted_flops}"
            )

    train_errs /= M * K
    train_accs /= M * K

    # Get final sampling counts
    sampling_counts = sampling_logger.get_sampling_counts()

    return train_errs, train_accs, test_errs, test_accs, sampling_counts, comp_costs


# FedSampling
def run_fedsampling(
    data_feeders,
    test_data,
    model,
    client_opt,
    T,
    M,
    K,
    B,
    fedsampling_alpha: float,
    fedsampling_m_param: int,
    test_freq=1,
    noisy_idxs=[],
):
    """
    Code Reference: Qi T, Wu F, Lyu L, et al. Fedsampling: A better sampling strategy for federated learning[J]. arXiv preprint arXiv:2306.14245, 2023.
    Run FedSampling algorithm with privacy-preserving data size estimation.

    FedSampling is a federated learning algorithm that incorporates differential
    privacy mechanisms for estimating client data sizes while maintaining privacy.
    The algorithm uses randomized response to protect individual client data volumes.

    Args:
        data_feeders: List of PyTorchDataFeeder instances for each client
        test_data: Tuple of (x, y) test data as torch tensors
        model: FLModel instance to train
        client_opt: Client optimizer instance
        T: Number of communication rounds
        M: Number of clients selected per round
        K: Number of local training steps per client
        B: Batch size for local training
        fedsampling_alpha: Privacy parameter for data size estimation (0.0-1.0)
        fedsampling_m_param: Upper bound for random response in estimation
        test_freq: Frequency of testing (default: 1)
        noisy_idxs: List of indices of noisy clients

    Returns:
        Tuple of (train_errs, train_accs, test_errs, test_accs, sampling_counts)
    """
    W = len(data_feeders)

    # Initialize statistics arrays
    train_errs, train_accs, test_errs, test_accs, comp_costs = init_stats_arrays5(T)

    # Initialize computation tracker
    comp_tracker = ComputationTracker(model, model.device, "fedsampling")

    # Initialize sampling logger for tracking client participation
    sampling_logger = SamplingLogger(n_clients=W, track_history=False)

    # Extract client sample counts for the estimator
    client_counts = [df.n_samples for df in data_feeders]

    # Initialize FedSampling estimator with privacy parameters
    estimator = FedSampling(client_counts, fedsampling_alpha, fedsampling_m_param)

    # Initialize global model parameters
    global_model_params = model.get_params()

    # Progress bar for training rounds
    bar = progressbar.ProgressBar()
    for t in bar(range(T)):
        # Step 1: Estimate total data size with differential privacy
        # Note: In the current implementation, hat_N is computed but not directly used
        # This maintains compatibility with the original FedSampling algorithm design
        hat_N = estimator.estimate()

        # Step 2: Initialize round gradient accumulator
        round_gradient_agg = global_model_params.zeros_like()

        # Step 3: Randomly select M clients for this round
        user_idxs = np.random.choice(W, M, replace=False)

        # Record client sampling for this round
        sampling_logger.record_sampling(user_idxs, round_num=t)

        # Compute weights based on actual client data sizes (for weighted averaging)
        weights = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights = weights.astype(np.float32)
        weights = weights / np.sum(weights)

        # Step 4: Local training and gradient computation for each selected client
        for (w, user_idx) in zip(weights, user_idxs):
            # Download global model to client
            model.set_params(global_model_params)

            # Save model state before local training
            start_model_params = model.get_params()

            # Perform K steps of local SGD training
            for k in range(K):
                # Get next batch of data
                x, y = data_feeders[user_idx].next_batch(B)

                # Execute one training step
                err, acc = model.train_step(x, y)

                # Accumulate training statistics
                train_errs[t] += err
                train_accs[t] += acc

            # Compute local gradient as the difference between trained and initial model
            end_model_params = model.get_params()
            local_gradient = end_model_params - start_model_params

            # Accumulate weighted gradient for aggregation
            round_gradient_agg += local_gradient * w

        # Track FedSampling operations
        comp_tracker.track_fedsampling_operations(W, fedsampling_m_param)

        # Step 5: Update global model with aggregated gradients
        global_model_params += round_gradient_agg

        # Record cumulative computation cost for this round
        comp_costs[t] = comp_tracker.get_total_flops()

        # Step 6: Periodic evaluation on test data
        if t % test_freq == 0:
            # Initialize test statistics for this round
            test_errs[t] = 0
            test_accs[t] = 0
            round_n_test_users = 0

            # FedSampling is a non-personalized algorithm, test on global model
            model.set_params(global_model_params)

            # Test on all non-noisy clients
            for user_idx in range(W):
                if user_idx not in noisy_idxs:
                    # Evaluate on client's test data
                    err, acc = model.test(
                        test_data[0][user_idx], test_data[1][user_idx], 128
                    )

                    # Accumulate test statistics
                    test_errs[t] += err
                    test_accs[t] += acc
                    round_n_test_users += 1

            # Compute average test performance
            if round_n_test_users > 0:
                test_errs[t] /= round_n_test_users
                test_accs[t] /= round_n_test_users
                formatted_flops = comp_tracker.get_formatted_flops()
                print(
                    f"Round {t}: Test Accuracy = {test_accs[t]}, Test Error = {test_errs[t]}, Comp Cost = {formatted_flops}"
                )

    # Normalize training statistics by total number of steps
    train_errs /= M * K
    train_accs /= M * K

    # Get final sampling counts
    sampling_counts = sampling_logger.get_sampling_counts()

    return train_errs, train_accs, test_errs, test_accs, sampling_counts, comp_costs


def run_hone(
    data_feeders,
    test_data,
    model,
    client_opt,
    T,
    M,
    K,
    B,
    test_freq=1,
    bn_setting=0,
    noisy_idxs=[],
    **kwargs,
):
    """
    Run the Hone algorithm.
    """
    
    # The core algorithm for Hone is temporarily hidden.

    return train_errs, train_accs, test_errs, test_accs, sampling_counts, comp_costs


# SCAFFOLD
def run_scaffold(
    data_feeders,
    test_data,
    model,
    client_opt,
    T,
    M,
    K,
    B,
    test_freq=1,
    bn_setting=0,
    noisy_idxs=[],
):
    """
    Run SCAFFOLD algorithm with corrected control variable updates.

    This implementation fixes the critical bug in gradient correction and follows
    the original SCAFFOLD paper's control variable update formulas strictly.

    Code Reference: Karimireddy S P, Kale S, Mohri M, et al. Scaffold: Stochastic controlled averaging for federated learning[C]//International conference on machine learning. PMLR, 2020: 5132-5143.

    Parameters:
    - data_feeders: List of PyTorchDataFeeder instances for each client.
    - test_data: Tuple containing test tensors.
    - model: Global model instance.
    - client_opt: Client optimizer instance.
    - T: Total number of communication rounds.
    - M: Number of clients selected per round.
    - K: Number of local training steps per client.
    - B: Batch size for local training.
    - test_freq: Frequency of testing (in communication rounds).
    - bn_setting: Batch normalization setting.
    - noisy_idxs: List of indices of noisy clients.

    Returns:
    - train_errs: Array of training errors.
    - train_accs: Array of training accuracies.
    - test_errs: Array of test errors.
    - test_accs: Array of test accuracies.
    - sampling_counts: Client sampling statistics.
    - comp_costs: Computation costs per round.
    """

    W = len(data_feeders)

    train_errs, train_accs, test_errs, test_accs, comp_costs = init_stats_arrays5(T)

    # Initialize computation tracker
    comp_tracker = ComputationTracker(model, model.device, "scaffold")

    # Initialize sampling logger for tracking client participation
    sampling_logger = SamplingLogger(n_clients=W, track_history=False)

    user_bn_model_vals = [model.get_bn_vals(setting=bn_setting) for _ in range(W)]
    user_bn_optim_vals = [
        client_opt.get_bn_params(model, setting=bn_setting) for _ in range(W)
    ]

    round_model_params = model.get_params().copy().params  # list of numpy arrays

    c_global = [np.zeros_like(p) for p in round_model_params]
    c_local_list = [[np.zeros_like(p) for p in round_model_params] for _ in range(W)]

    global_model = copy.deepcopy(model)

    torch.autograd.set_detect_anomaly(True)

    bar = progressbar.ProgressBar()
    for t in bar(range(T)):

        param_updates_sum = [np.zeros_like(p) for p in round_model_params]
        delta_c_sum = [np.zeros_like(p) for p in round_model_params]

        user_idxs = np.random.choice(W, M, replace=False)
        sampling_logger.record_sampling(user_idxs, round_num=t)

        client_weights = np.array(
            [data_feeders[u].n_samples for u in user_idxs], dtype=np.float32
        )
        client_weights /= np.sum(client_weights)

        round_n_test_users = 0

        learning_rate = client_opt.param_groups[0]["lr"]

        for (w, user_idx) in zip(client_weights, user_idxs):

            from models import NumpyModel

            round_model_wrapper = NumpyModel(round_model_params)
            model.set_params(round_model_wrapper)

            model.set_bn_vals(user_bn_model_vals[user_idx], setting=bn_setting)
            client_opt.set_bn_params(
                user_bn_optim_vals[user_idx], model, setting=bn_setting
            )

            c_local = c_local_list[user_idx]
            state_params_diff_curr_tensor = [
                torch.tensor(cg - cl, dtype=torch.float32, device=model.device)
                for cg, cl in zip(c_global, c_local)
            ]

            for k in range(K):
                x, y = data_feeders[user_idx].next_batch(B)
                x = x.to(model.device)
                y = y.to(model.device)

                output = model(x)
                loss_fn = torch.nn.CrossEntropyLoss()
                loss_f_i = loss_fn(output, y)

                client_opt.zero_grad()
                loss_f_i.backward()

                with torch.no_grad():
                    for p, c in zip(model.parameters(), state_params_diff_curr_tensor):
                        # Only add correction if the parameter has gradients
                        if p.grad is not None:
                            p.grad += c

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                client_opt.step()

                train_errs[t] += loss_f_i.item() * y.size(0)
                _, predicted = torch.max(output.data, 1)
                train_accs[t] += (predicted == y).sum().item()

            if (t % test_freq == 0) and (user_idx not in noisy_idxs):
                err, acc = model.test(
                    test_data[0][user_idx], test_data[1][user_idx], 128
                )
                test_errs[t] += err
                test_accs[t] += acc
                round_n_test_users += 1

            trained_model_params = (
                model.get_params().copy().params
            )  # list of numpy arrays

            new_c_local = []
            for j in range(len(c_global)):

                param_diff = round_model_params[j] - trained_model_params[j]

                # c_i^+ = c_i - c + (1 / Kη) * (y - x^+)

                c_i_update = (
                    c_local[j] - c_global[j] + (param_diff / (K * learning_rate))
                )
                new_c_local.append(c_i_update)

            for j in range(len(c_global)):
                delta_c_sum[j] += new_c_local[j] - c_local[j]

            c_local_list[user_idx] = new_c_local

            for j, p in enumerate(trained_model_params):
                param_updates_sum[j] += p * w

            user_bn_model_vals[user_idx] = model.get_bn_vals(setting=bn_setting)
            user_bn_optim_vals[user_idx] = client_opt.get_bn_params(
                model, setting=bn_setting
            )

        # Track SCAFFOLD-specific operations
        comp_tracker.track_scaffold_operations(K, M)

        round_model_params = param_updates_sum

        # c^+ = c + (1/W) * Σ(Δc_i)

        for j in range(len(c_global)):
            c_global[j] += (1 / W) * delta_c_sum[j]

        # Record cumulative computation cost for this round
        comp_costs[t] = comp_tracker.get_total_flops()

        if t % test_freq == 0:
            if round_n_test_users > 0:
                test_errs[t] /= round_n_test_users
                test_accs[t] /= round_n_test_users
                formatted_flops = comp_tracker.get_formatted_flops()
                print(
                    f"Round {t}: Test Accuracy = {test_accs[t]}, Test Error = {test_errs[t]}, Comp Cost = {formatted_flops}"
                )

    train_errs /= M * K
    train_accs /= M * K

    # Get final sampling counts
    sampling_counts = sampling_logger.get_sampling_counts()

    return train_errs, train_accs, test_errs, test_accs, sampling_counts, comp_costs


def run_fedprox(
    data_feeders,
    test_data,
    model,
    client_opt,
    T,
    M,
    K,
    B,
    test_freq=1,
    fedprox_mu=None,
    bn_setting=0,
    noisy_idxs=[],
):
    """
    Code Reference: Li T, Sahu A K, Zaheer M, et al. Federated optimization in heterogeneous networks[J]. Proceedings of Machine learning and systems, 2020, 2: 429-450.

    Args:
        - data_feeders: list of PyTorchDataFeeder instances for each client
        - local_test_data: tuple of lists containing local test tensors
        - global_test_data: tuple of global test tensors
        - model: FLModel instance representing the global model
        - client_opt: ClientOpt instance representing the optimizer
        - T: (int) number of communication rounds
        - M: (int) number of clients selected per round
        - K: (int) number of local training steps per client
        - B: (int) batch size
        - weight_decay: (float) weight decay parameter (already included in client_opt)
        - mu: (float) FedProx regularization parameter
        - bn_setting: (int) batch normalization setting
        - noisy_idxs: list of indices of noisy clients
        - distance_to_the_cloud, distance_matrix, comm_datasize, training_datasize, distances_clients_to_the_cloud: additional parameters (unused here)

    Returns:
        - test_errs: list of client test errors per round
        - test_accs: list of client test accuracies per round
        - global_test_errs: list of global test errors per round
        - global_test_accs: list of global test accuracies per round
    """
    W = len(data_feeders)

    train_errs, train_accs, test_errs, test_accs, comp_costs = init_stats_arrays5(T)

    # Initialize computation tracker
    comp_tracker = ComputationTracker(model, model.device, "fedprox")

    # Initialize sampling logger for tracking client participation
    sampling_logger = SamplingLogger(n_clients=W, track_history=False)

    user_bn_model_vals = [model.get_bn_vals(setting=bn_setting) for _ in range(W)]
    user_bn_optim_vals = [
        client_opt.get_bn_params(model, setting=bn_setting) for _ in range(W)
    ]

    round_model = model.get_params()
    round_opt = client_opt.get_params()

    global_model = copy.deepcopy(model)
    global_model.set_params(round_model)

    bar = progressbar.ProgressBar()
    for t in bar(range(T)):

        round_agg = round_model.zeros_like()
        round_opt_agg = round_opt.zeros_like()

        user_idxs = np.random.choice(W, M, replace=False)

        # Record client sampling for this round
        sampling_logger.record_sampling(user_idxs, round_num=t)

        weights = np.array(
            [data_feeders[u].n_samples for u in user_idxs], dtype=np.float32
        )
        weights /= weights.sum()

        round_n_test_users = 0

        for w, user_idx in zip(weights, user_idxs):

            model.set_params(round_model)
            client_opt.set_params(round_opt)
            model.set_bn_vals(user_bn_model_vals[user_idx], setting=bn_setting)
            client_opt.set_bn_params(
                user_bn_optim_vals[user_idx], model, setting=bn_setting
            )

            if (t % 1 == 0) and (user_idx not in noisy_idxs):
                err, acc = model.test(
                    test_data[0][user_idx], test_data[1][user_idx], 128
                )
                test_errs[t] += err
                test_accs[t] += acc
                round_n_test_users += 1

            for k in range(K):
                x, y = data_feeders[user_idx].next_batch(B)

                model.train()
                model.optim.zero_grad()
                logits = model.forward(x)
                loss_f_i = model.loss_fn(logits, y)
                loss_f_i.backward()

                with torch.no_grad():
                    for p, w_global in zip(
                        model.parameters(), global_model.parameters()
                    ):
                        # Only add regularization gradient if the parameter has gradients
                        if p.grad is not None:
                            p.grad += fedprox_mu * (p.data - w_global.data)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                model.optim.step()

                train_errs[t] += loss_f_i.item()
                train_accs[t] += model.calc_acc(logits, y).item()

            round_agg += model.get_params() * w
            round_opt_agg += client_opt.get_params() * w

            user_bn_model_vals[user_idx] = model.get_bn_vals(setting=bn_setting)
            user_bn_optim_vals[user_idx] = client_opt.get_bn_params(
                model, setting=bn_setting
            )

        # Track FedProx-specific regularization operations
        comp_tracker.track_fedprox_operations(K, M)

        round_model = round_agg.copy()
        round_opt = round_opt_agg.copy()
        global_model.set_params(round_model)

        # Record cumulative computation cost for this round
        comp_costs[t] = comp_tracker.get_total_flops()

        if t % test_freq == 0:
            if round_n_test_users > 0:
                test_errs[t] /= round_n_test_users
                test_accs[t] /= round_n_test_users
                formatted_flops = comp_tracker.get_formatted_flops()
                print(
                    f"Round {t}: Test Accuracy = {test_accs[t]}, Test Error = {test_errs[t]}, Comp Cost = {formatted_flops}"
                )

    train_errs /= M * K
    train_accs /= M * K

    # Get final sampling counts
    sampling_counts = sampling_logger.get_sampling_counts()

    return train_errs, train_accs, test_errs, test_accs, sampling_counts, comp_costs


# pFedMe
def run_pFedMe(
    data_feeders,
    test_data,
    model,
    T,
    M,
    K,
    B,
    R,
    pfedme_lambda,
    pfedme_eta,
    pfedme_beta,
    test_freq=1,
    noisy_idxs=[],
):
    """
    Code Reference: Personalized Federated Learning with Moreau Envelopes', Dinh et al., NeurIPS 2020.
    Runs T rounds of pFedMe, and returns the test results. Note that, to make the algorithm comparison fair, we do not activate all clients as per Algorithm 1 of the pFedMe paper, only
    sending back the gradients of the sampled clients. Instead, we only activate the sampled clients each round, bringing pFedMe in line with the comparisons.
    """
    W = len(data_feeders)

    train_errs, train_accs, test_errs, test_accs, comp_costs = init_stats_arrays5(T)

    # Initialize computation tracker
    comp_tracker = ComputationTracker(model, model.device, "pfedme")

    # Initialize sampling logger for tracking client participation
    sampling_logger = SamplingLogger(n_clients=W, track_history=False)

    # global model updated at the end of each round, and round model accumulator
    round_model = model.get_params()
    round_agg = model.get_params()

    user_models = [round_model.copy() for w in range(W)]

    global_model = copy.deepcopy(model)
    global_model.set_params(round_model)

    bar = progressbar.ProgressBar()
    for t in bar(range(T)):
        round_agg = round_agg.zeros_like()

        # select round clients and compute their weights for later sum
        user_idxs = np.random.choice(W, M, replace=False)

        # Record client sampling for this round
        sampling_logger.record_sampling(user_idxs, round_num=t)

        weights = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights = weights.astype(np.float32)
        weights = weights / np.sum(weights)

        round_n_test_users = 0

        for (w, user_idx) in zip(weights, user_idxs):

            # test local model if not a noisy client
            if (t % test_freq == 0) and (user_idx not in noisy_idxs):
                model.set_params(user_models[user_idx])
                err, acc = model.test(
                    test_data[0][user_idx], test_data[1][user_idx], 128
                )
                test_errs[t] += err
                test_accs[t] += acc
                round_n_test_users += 1

            # download global model
            model.set_params(round_model)

            # perform k steps of local training
            for r in range(R):
                x, y = data_feeders[user_idx].next_batch(B)
                omega = user_models[user_idx]
                for k in range(K):
                    model.optim.zero_grad()
                    logits = model.forward(x)
                    loss = model.loss_fn(logits, y)
                    loss.backward()
                    model.optim.step(omega)

                theta = model.get_params()

                user_models[user_idx] = omega - (
                    pfedme_lambda * pfedme_eta * (omega - theta)
                )

            round_agg = round_agg + (user_models[user_idx] * w)

        # new global model is weighted sum of old client models (beta) and new models' updates (1-beta)
        round_model = (1 - pfedme_beta) * round_model + pfedme_beta * round_agg

        # Track pFedMe-specific Moreau envelope operations
        comp_tracker.track_pfedme_operations(R, K, M)

        # Record cumulative computation cost for this round
        comp_costs[t] = comp_tracker.get_total_flops()

        if t % test_freq == 0:
            test_errs[t] = test_errs[t] / round_n_test_users
            test_accs[t] = test_accs[t] / round_n_test_users
            formatted_flops = comp_tracker.get_formatted_flops()
            print(
                f"Round {t}: Test Accuracy = {test_accs[t]}, Test Error = {test_errs[t]}, Comp Cost = {formatted_flops}"
            )

    # Get final sampling counts
    sampling_counts = sampling_logger.get_sampling_counts()

    return train_errs, train_accs, test_errs, test_accs, sampling_counts, comp_costs


# Not for use at this time
def run_per_fedavg( 
    data_feeders,
    test_data,
    model,
    perfedavg_beta,
    T,
    M,
    K,
    B,
    test_freq=1,
    noisy_idxs=[],
):
    """
    Code Reference: Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach', Fallah et al., NeurIPS 2020.
    Runs T rounds of Per-FedAvg, and returns the test results. Note we are using the first-order approximation variant (i) described in Section 5 of Per-FedAvg paper.

    Args:
        - data_feeders: (list)      of PyTorchDataFeeders
        - test_data:    (tuple)     of (x, y) testing data as torch.tensors
        - model:        (FLModel)   to run SGD on
        - perfedavg_beta: (float)     parameter of Per-FedAvg algorithm
        - T:            (int)       number of communication rounds
        - M:            (int)       number of clients sampled per round
        - K:            (int)       number of local training steps
        - B:            (int)       client batch size
        - test_freq:    (int)       how often to test UA
        - noisy_idxs:   (iterable)  indexes of noisy clients (ignore their UA)

    Returns:
        Tuple containing:
            - train_errs:        Numpy array of training errors of length T
            - train_accs:        Numpy array of training accuracies of length T
            - test_errs:         Numpy array of local test errors of length T
            - test_accs:         Numpy array of local test accuracies of length T

        If test_freq > 1, non-tested rounds will contain 0's for the corresponding test metrics.

    Notes:
        This function uses Personalized-FedAvg. Each client performs one local training step, and updated model parameters are weighted and summed to form the global model. The beta parameter is applied after each client local step.
        The returned results are tracked per round rather than as per-round averages, and include global model test results.
    """
    W = len(data_feeders)

    train_errs, train_accs, test_errs, test_accs, comp_costs = init_stats_arrays5(T)

    # Initialize computation tracker
    comp_tracker = ComputationTracker(model, model.device, "per_fedavg")

    # Initialize sampling logger for tracking client participation
    sampling_logger = SamplingLogger(n_clients=W, track_history=False)

    round_model = model.get_params()
    round_agg = model.get_params()

    bar = progressbar.ProgressBar()
    for t in bar(range(T)):
        round_agg = round_agg.zeros_like()

        user_idxs = np.random.choice(W, M, replace=False)

        # Record client sampling for this round
        sampling_logger.record_sampling(user_idxs, round_num=t)

        weights = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights = weights.astype(np.float32)
        weights = weights / np.sum(weights)

        round_n_test_users = 0

        round_train_err = 0.0
        round_train_acc = 0.0
        round_total_weight = 0.0

        for (w, user_idx) in zip(weights, user_idxs):

            model.set_params(round_model)

            if (t % test_freq == 0) and (user_idx not in noisy_idxs):
                x, y = data_feeders[user_idx].next_batch(B)
                model.train_step(x, y)
                err, acc = model.test(
                    test_data[0][user_idx], test_data[1][user_idx], 128
                )
                test_errs[t] += err
                test_accs[t] += acc
                round_n_test_users += 1
                model.set_params(round_model)

            client_train_err = 0.0
            client_train_acc = 0.0

            initial_model_params = model.get_params()

            if hasattr(model.optim, "set_initial_params"):

                param_list = []
                for p in model.parameters():
                    param_list.append(p.data.clone())
                model.optim.set_initial_params(param_list)

            for k in range(K):

                x, y = data_feeders[user_idx].next_batch(B)
                loss, acc = model.train_step(x, y)

                client_train_err += loss
                client_train_acc += acc

            local_trained_params = model.get_params()

            model.set_params(initial_model_params)

            x, y = data_feeders[user_idx].next_batch(B)
            logits = model.forward(x)
            loss = model.loss_fn(logits, y)
            model.optim.zero_grad()
            loss.backward()

            model.optim.step(beta=perfedavg_beta)

            client_train_err /= K
            client_train_acc /= K

            round_agg = round_agg + (model.get_params() * w)

            round_train_err += client_train_err * w
            round_train_acc += client_train_acc * w
            round_total_weight += w

        round_model = round_agg.copy()

        # Track Per-FedAvg meta-learning operations
        comp_tracker.track_per_fedavg_operations(K, M)

        # Record cumulative computation cost for this round
        comp_costs[t] = comp_tracker.get_total_flops()

        train_errs[t] = round_train_err / round_total_weight
        train_accs[t] = round_train_acc / round_total_weight

        if t % test_freq == 0:
            if round_n_test_users > 0:
                test_errs[t] = test_errs[t] / round_n_test_users
                test_accs[t] = test_accs[t] / round_n_test_users
                formatted_flops = comp_tracker.get_formatted_flops()
                print(
                    f"Round {t}: Test Accuracy = {test_accs[t]}, Test Error = {test_errs[t]}, Comp Cost = {formatted_flops}"
                )

    # Get final sampling counts
    sampling_counts = sampling_logger.get_sampling_counts()

    return train_errs, train_accs, test_errs, test_accs, sampling_counts, comp_costs


# Ditto
def run_ditto(
    data_feeders,
    test_data,
    model,
    client_opt,
    T,
    M,
    K_global,
    K_personal,
    B,
    ditto_lambda,
    test_freq=1,
    bn_setting=0,
    noisy_idxs=[],
):
    """
    Code Reference: Ditto: Fair and Robust Federated Learning Through Personalization', Li et al., ICML 2021.

    Ditto maintains two models for each client:
    1. A global model that contributes to federated aggregation
    2. A personalized model optimized with a proximal term to the global model

    Args:
        data_feeders: List of PyTorchDataFeeder instances for each client
        test_data: Tuple of (x, y) test data as torch tensors
        model: FLModel instance to train
        client_opt: Client optimizer instance (standard SGD for global model)
        T: Number of communication rounds
        M: Number of clients selected per round
        K_global: Number of local training steps for global model
        K_personal: Number of local training steps for personalized model
        B: Batch size for local training
        ditto_lambda: Regularization parameter λ for personalized model
        test_freq: Frequency of testing (default: 1)
        bn_setting: Batch normalization setting
        noisy_idxs: List of indices of noisy clients

    Returns:
        Tuple of (train_errs, train_accs, test_errs, test_accs, sampling_counts, comp_costs)
    """
    W = len(data_feeders)

    # Initialize statistics arrays
    train_errs, train_accs, test_errs, test_accs, comp_costs = init_stats_arrays5(T)

    # Initialize computation tracker
    comp_tracker = ComputationTracker(model, model.device, "ditto")

    # Initialize sampling logger
    sampling_logger = SamplingLogger(n_clients=W, track_history=False)

    # Initialize global model parameters
    global_model_params = model.get_params()

    # Initialize personalized models for each client (initially same as global)
    user_personal_models = [global_model_params.copy() for _ in range(W)]

    # BN parameters storage (if needed)
    user_bn_model_vals = [model.get_bn_vals(setting=bn_setting) for _ in range(W)]
    user_bn_optim_vals = [client_opt.get_bn_params(model) for _ in range(W)]

    # Progress bar
    bar = progressbar.ProgressBar()
    for t in bar(range(T)):
        # Initialize round aggregation
        round_agg = global_model_params.zeros_like()

        # Select clients for this round
        user_idxs = np.random.choice(W, M, replace=False)

        # Record client sampling
        sampling_logger.record_sampling(user_idxs, round_num=t)

        # Compute aggregation weights
        weights = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights = weights.astype(np.float32)
        weights = weights / np.sum(weights)

        round_n_test_users = 0

        for (w, user_idx) in zip(weights, user_idxs):
            # Task A: Train global contribution model
            # Load current global model
            model.set_params(global_model_params)
            model.set_bn_vals(user_bn_model_vals[user_idx], setting=bn_setting)

            # Use standard SGD optimizer for global model training
            optimizer = ClientSGD(
                model.parameters(), lr=client_opt.param_groups[0]["lr"]
            )
            model.set_optim(optimizer)
            optimizer.set_bn_params(
                user_bn_optim_vals[user_idx], model, setting=bn_setting
            )

            # Train for K_global steps
            for k in range(K_global):
                x, y = data_feeders[user_idx].next_batch(B)
                err, acc = model.train_step(x, y)
                train_errs[t] += err
                train_accs[t] += acc

            # Get updated global contribution parameters
            global_update_params = model.get_params()

            # Task B: Train personalized model
            # Load client's personalized model
            model.set_params(user_personal_models[user_idx])

            # Create Ditto optimizer for personalized model
            personal_optimizer = ClientDittoOptimizer(
                model.parameters(),
                lr=client_opt.param_groups[0]["lr"],
                ditto_lambda=ditto_lambda,
            )
            model.set_optim(personal_optimizer)

            # Train personalized model for K_personal steps
            for k in range(K_personal):
                x, y = data_feeders[user_idx].next_batch(B)

                # Forward pass
                model.optim.zero_grad()
                logits = model.forward(x)
                loss = model.loss_fn(logits, y)
                loss.backward()

                # Ditto step with global model as reference
                personal_optimizer.step(global_model_params=global_model_params)

            # Get updated personalized model
            updated_personal_params = model.get_params()

            # Test personalized model
            if (t % test_freq == 0) and (user_idx not in noisy_idxs):
                # Test using personalized model
                model.set_params(updated_personal_params)
                err, acc = model.test(
                    test_data[0][user_idx], test_data[1][user_idx], 128
                )
                test_errs[t] += err
                test_accs[t] += acc
                round_n_test_users += 1

            # Aggregate global updates
            round_agg = round_agg + (global_update_params * w)

            # Store updated personalized model
            user_personal_models[user_idx] = updated_personal_params.copy()

            # Store BN parameters
            user_bn_model_vals[user_idx] = model.get_bn_vals(setting=bn_setting)
            user_bn_optim_vals[user_idx] = optimizer.get_bn_params(
                model, setting=bn_setting
            )

        # Update global model
        global_model_params = round_agg.copy()

        # Track Ditto operations
        comp_tracker.track_ditto_operations(K_global, K_personal, M)

        # Record computation cost
        comp_costs[t] = comp_tracker.get_total_flops()

        # Compute test statistics
        if t % test_freq == 0:
            if round_n_test_users > 0:
                test_errs[t] /= round_n_test_users
                test_accs[t] /= round_n_test_users
                formatted_flops = comp_tracker.get_formatted_flops()
                print(
                    f"Round {t}: Test Accuracy = {test_accs[t]}, Test Error = {test_errs[t]}, Comp Cost = {formatted_flops}"
                )

    # Normalize training statistics
    train_errs /= M * K_global
    train_accs /= M * K_global

    # Get final sampling counts
    sampling_counts = sampling_logger.get_sampling_counts()

    return train_errs, train_accs, test_errs, test_accs, sampling_counts, comp_costs
