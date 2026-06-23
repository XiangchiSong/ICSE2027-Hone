import os
import numpy as np
import pickle
import torch
from parameterParse import parse_args
from data_utils import *
from models import *
from fl_optimisers import *
from fl_algs import *

# GPU Performance Optimization Settings
def setup_gpu_optimization(enable_deterministic=True):
    """
    Configure GPU settings for optimal performance while maintaining reproducibility.

    Args:
        enable_deterministic: If True, enables deterministic algorithms for reproducibility
                            If False, prioritizes performance over strict reproducibility
    """
    if enable_deterministic:
        # Required for pytorch deterministic GPU behaviour (slower but reproducible)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("INFO: Deterministic algorithms enabled (slower but reproducible)")
    else:
        # PERFORMANCE MODE: Remove all GPU performance restrictions
        # Remove CUBLAS workspace restrictions for maximum performance
        if "CUBLAS_WORKSPACE_CONFIG" in os.environ:
            del os.environ["CUBLAS_WORKSPACE_CONFIG"]

        # Disable deterministic algorithms for maximum speed
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner

        # Additional performance optimizations
        torch.backends.cudnn.enabled = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = (
                True  # Enable TensorFloat-32 for faster training
            )
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = (
                True  # Enable TF32 for matrix operations
            )

        print("🚀 INFO: PERFORMANCE MODE ACTIVATED")
        print("   - Deterministic algorithms: DISABLED")
        print("   - CUBLAS workspace restrictions: REMOVED")
        print("   - cuDNN benchmark mode: ENABLED")
        print("   - TensorFloat-32: ENABLED (if available)")

    # GPU memory optimization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"INFO: GPU ready - {device_name} ({memory_gb:.1f}GB)")
    else:
        print("WARNING: CUDA not available, running on CPU")


def get_fname(a):
    """
    Args:
        - a: (argparse.Namespace) command-line arguments

    Returns:
        Underscore-separated str ending with '.pkl', containing items in args.
    """
    # Parameters that are fixed and should not appear in filename
    excluded_params = {
        "device",  # Fixed to 'gpu' for all algorithms
    }

    # Additional parameters fixed specifically for hone algorithm
    if hasattr(a, "alg") and a.alg == "hone":
        excluded_params.update(
            {
                "bn_private",  # Fixed to 'none'
                "c_d",  # Fixed to 200.0
                "epsilon_d",  # Fixed to 10.0
                "c_g",  # Fixed to 10.0
                "c_l",  # Fixed to 20.0
                "epsilon_contrib",  # Fixed to 50.0
                "delta_coalition",  # Fixed to 0.2
                "gamma1",  # Fixed to 0.9
                "gamma2",  # Fixed to 0.8
                "T_warmup",  # Automatically calculated as T//10
            }
        )

    # Additional parameters fixed specifically for page algorithm
    if hasattr(a, "alg") and a.alg == "page":
        excluded_params.update(
            {
                "page_warmup",  # Fixed to 35
                "page_reward_alpha",  # Fixed to 0.5
                "page_c_hidden1",  # Fixed to 40
                "page_c_hidden2",  # Fixed to 30
                "page_c_actor_lr",  # Fixed to 1e-4
                "page_c_critic_lr",  # Fixed to 1e-3
                "page_s_hidden1",  # Fixed to 40
                "page_s_hidden2",  # Fixed to 30
                "page_s_actor_lr",  # Fixed to 1e-4
                "page_s_critic_lr",  # Fixed to 1e-3
            }
        )

    if hasattr(a, "alg") and a.alg == "fedsampling":
        excluded_params.update(
            {
                "fedsampling_alpha",  # Fixed to 0.5
                "fedsampling_m_param",  # Fixed to 20
            }
        )

    if hasattr(a, "alg") and a.alg == "fedala":
        excluded_params.update(
            {
                "fedala_rand_percent",  # Fixed to 10
                "fedala_layer_idx",  # Fixed to 0
                "fedala_threshold",  # Fixed to 0.1
                "fedala_num_pre_loss",  # Fixed to 10
            }
        )

    fname = "_".join(
        [
            k + "-" + str(v)
            for (k, v) in vars(a).items()
            if not v is None and k not in excluded_params
        ]
    )
    return fname + ".pkl"


def save_data(data, fname):
    """
    Saves data in pickle format.

    Args:
        - data:  (object)   to save
        - fname: (str)      file path to save to
    """
    with open(fname, "wb") as f:
        pickle.dump(data, f)


def load_pkl_file(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def main():
    """
    Run experiment specified by command-line args.
    """
    args = parse_args()

    # GPU optimization setup - PERFORMANCE MODE ENABLED
    # Set to False for maximum performance, True for reproducible results
    enable_deterministic = False  # PERFORMANCE MODE: Faster training, less reproducible
    setup_gpu_optimization(enable_deterministic)

    global_seed = args.seed
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(global_seed)
        torch.cuda.manual_seed_all(global_seed)

    device = torch.device(
        "cuda:0" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    )
    print(f"INFO: Using device: {device}")

    # load data
    print("Loading and partitioning data...")
    data_path = f"../{args.dset.upper()}_data"

    # Check if dataset is MIND and use specialized loader
    if args.dset == "mind":
        # Load MIND dataset with specialized function
        train, test, embedding_matrix, num_classes = load_mind_federated_dataset(
            data_dir=data_path,
            num_clients=args.W,
            iid_label=args.iid_label,
            iid_data=args.iid_data,
            user_test=True,
            seed=args.seed,
        )
    else:
        # Load other datasets with standard function
        train, test = load_federated_dataset(
            dataset_name=args.dset,
            data_dir=data_path,
            num_clients=args.W,
            iid_label=args.iid_label,
            iid_data=args.iid_data,
            user_test=True,  # Keep True to match current algorithm's needs for local test sets
            seed=args.seed,  # Use the same seed for reproducibility
            test_fold=args.test_fold,  # For UrbanSound8K 10-fold CV support
        )

    # Set dataset-specific parameters with GPU-optimized batch size recommendations
    if args.dset == "mnist":
        model = MNISTModel(device)
        noise_std = 3.0
        steps_per_E = int(np.round(60000 / (args.W * args.B)))
        recommended_batch_sizes = [64, 128, 256, 512]

    elif args.dset == "cifar10":
        model = CIFAR10Model(device)
        noise_std = 0.2
        steps_per_E = int(np.round(50000 / (args.W * args.B)))
        recommended_batch_sizes = [64, 128, 256]

    elif args.dset == "cifar100":
        model = CIFAR100Model(device)
        noise_std = 0.26
        steps_per_E = int(np.round(50000 / (args.W * args.B)))
        recommended_batch_sizes = [64, 128, 256]

    elif args.dset == "tinyimagenet":
        model = TinyImageNetModel(device)
        noise_std = 0.2
        steps_per_E = int(np.round(100000 / (args.W * args.B)))
        recommended_batch_sizes = [32, 64, 128, 256]  # ResNet-50 is memory intensive

    elif args.dset == "mind":
        # embedding_matrix and num_classes are already loaded from load_mind_specific_data
        model = MINDModel(
            device, embedding_matrix=embedding_matrix, num_classes=num_classes
        )
        noise_std = 0.0  # No noise for text data
        # Estimate steps per epoch based on typical MIND dataset size
        steps_per_E = int(np.round(50000 / (args.W * args.B)))  # Approximate size
        recommended_batch_sizes = [32, 64, 128]  # Text-CNN is memory efficient

    elif args.dset == "gtsrb":
        model = GTSRBModel(device)
        noise_std = 0.2  # Similar to CIFAR10
        # GTSRB training set has approximately 39,209 images
        steps_per_E = int(np.round(39209 / (args.W * args.B)))
        recommended_batch_sizes = [
            32,
            64,
            128,
        ]  # ResNet-34 is moderately memory intensive

    elif args.dset == "urbansound8k":
        model = UrbanSound8KModel(device)
        noise_std = 0.0  # Spectrograms don't need pixel noise
        # UrbanSound8K has 8732 total samples
        steps_per_E = int(np.round(8732 / (args.W * args.B)))
        recommended_batch_sizes = [16, 32, 64]  # Audio models may use more memory

    # Batch size optimization warning
    if args.B < 32:
        print(
            f"⚠️ WARNING: Current batch size ({args.B}) is very small for GPU training!"
        )
        print(f"   Recommended batch sizes for {args.dset}: {recommended_batch_sizes}")
        print(
            f"   Consider using -B {recommended_batch_sizes[1]} or -B {recommended_batch_sizes[2]} for better GPU utilization"
        )
    elif args.B >= 32:
        print(f"✅ INFO: Batch size {args.B} is suitable for GPU training")

    # add noise to data
    noisy_imgs, noisy_idxs = add_noise_to_frac(train[0], args.noisy_frac, noise_std)
    train = (noisy_imgs, train[1])

    # create data feeders, convert to pytorch tensors
    if args.dset == "mind":
        # For MIND dataset, x data should be 'long' type (token indices)
        feeders = [
            PyTorchDataFeeder(x, "long", y, "long", device)
            for (x, y) in zip(train[0], train[1])
        ]
        test_data = (
            [to_tensor(x, device, "long") for x in test[0]],
            [to_tensor(y, device, "long") for y in test[1]],
        )
    else:
        # For other datasets, x data is float32
        feeders = [
            PyTorchDataFeeder(x, torch.float32, y, "long", device)
            for (x, y) in zip(train[0], train[1])
        ]
        test_data = (
            [to_tensor(x, device, torch.float32) for x in test[0]],
            [to_tensor(y, device, "long") for y in test[1]],
        )

    # miscellaneous settings
    fname = get_fname(args)
    M = int(args.W * args.C)
    K = steps_per_E * args.E
    str_to_bn_setting = {"usyb": 0, "yb": 1, "us": 2, "none": 3}
    if args.alg in [
        "fedavg",
        "scaffold",
        "fedprox",
        "hone",
        "clusteredsampling",
        "page",
        "fedala",
        "ditto",
    ]:
        bn_setting = str_to_bn_setting[args.bn_private]

    # run experiment

    print("Starting experiment...")

    if args.alg == "fedavg":
        client_optim = ClientSGD(model.parameters(), lr=args.lr)
        model.set_optim(client_optim)
        data = run_fedavg(
            feeders,
            test_data,
            model,
            client_optim,
            args.T,
            M,
            K,
            args.B,
            bn_setting=bn_setting,
            noisy_idxs=noisy_idxs,
        )

    elif args.alg == "scaffold":
        client_optim = ClientSGD(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        model.set_optim(client_optim)
        data = run_scaffold(
            feeders,
            test_data,
            model,
            client_optim,
            args.T,
            M,
            K,
            args.B,
            bn_setting=bn_setting,
            noisy_idxs=noisy_idxs,
        )

    elif args.alg == "fedprox":
        client_optim = ClientSGD(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        model.set_optim(client_optim)
        data = run_fedprox(
            feeders,
            test_data,
            model,
            client_optim,
            args.T,
            M,
            K,
            args.B,
            fedprox_mu=args.fedprox_mu,
            bn_setting=bn_setting,
            noisy_idxs=noisy_idxs,
        )

    elif args.alg == "perfedavg":
        client_optim = ClientSGD_perfedavg(model.parameters(), lr=args.lr)
        model.set_optim(client_optim, init_optim=False)
        data = run_per_fedavg(
            feeders,
            test_data,
            model,
            args.perfedavg_beta,
            args.T,
            M,
            K,
            args.B,
            noisy_idxs=noisy_idxs,
        )

    elif args.alg == "pfedme":
        client_optim = pFedMeOptimizer(
            model.parameters(),
            device,
            pfedme_lr=args.lr,
            pfedme_lambda=args.pfedme_lambda,
        )
        model.set_optim(client_optim, init_optim=False)
        data = run_pFedMe(
            feeders,
            test_data,
            model,
            args.T,
            M,
            K=1,
            B=args.B,
            R=K,
            pfedme_lambda=args.pfedme_lambda,
            pfedme_eta=args.lr,
            pfedme_beta=args.pfedme_beta,
            noisy_idxs=noisy_idxs,
        )

    elif args.alg == "hone":
        client_optim = ClientSGD(model.parameters(), lr=args.lr)
        model.set_optim(client_optim)

        # Automatically set T_warmup to 1/10 of total rounds T
        args.T_warmup = max(1, args.T // 10)  # Ensure at least 1 warmup round
        print(
            f"INFO: T_warmup automatically set to {args.T_warmup} (T/10 = {args.T}/10)"
        )

        # Pack all args into a dictionary to pass to run_hone
        # This avoids a very long function signature and keeps it clean
        kwargs = {"args": args}
        data = run_hone(
            # The function call interface is temporarily hidden.
        )

    elif args.alg == "fedsampling":
        client_optim = ClientSGD(model.parameters(), lr=args.lr)
        model.set_optim(client_optim)
        data = run_fedsampling(
            feeders,
            test_data,
            model,
            client_optim,
            args.T,
            M,
            K,
            args.B,
            fedsampling_alpha=args.fedsampling_alpha,
            fedsampling_m_param=args.fedsampling_m_param,
            noisy_idxs=noisy_idxs,
        )

    elif args.alg == "clusteredsampling":
        print("Running Clustered Sampling (Algorithm 1)...")
        client_optim = ClientSGD(model.parameters(), lr=args.lr)
        model.set_optim(client_optim)
        data = run_clusteredsampling(
            feeders,
            test_data,
            model,
            client_optim,
            args.T,
            M,
            K,
            args.B,
            bn_setting=bn_setting,
            noisy_idxs=noisy_idxs,
        )

    elif args.alg == "page":
        print("Running PAGE (Personalized and Generalized balancing Game)...")
        client_optim = ClientSGD(model.parameters(), lr=args.lr)
        model.set_optim(client_optim)
        data = run_page(
            feeders,
            test_data,
            model,
            client_optim,
            args.T,
            M,
            K,
            args.B,
            args,
            bn_setting=bn_setting,
            noisy_idxs=noisy_idxs,
        )

    elif args.alg == "fedala":
        print("Running FedALA (Adaptive Local Aggregation)...")

        client_optim = ClientSGD(model.parameters(), lr=args.lr)
        model.set_optim(client_optim)
        data = run_fedala(
            feeders,
            test_data,
            model,
            client_optim,
            args.T,  # Total rounds
            M,  # Clients per round
            K,  # Local steps
            args.B,  # Batch size
            args,  # Pass all args for FedALA specific params
            bn_setting=bn_setting,
            noisy_idxs=noisy_idxs,
        )

    elif args.alg == "ditto":
        print("Running Ditto (Fair and Robust FL Through Personalization)...")

        client_optim = ClientSGD(model.parameters(), lr=args.lr)
        model.set_optim(client_optim)

        # Set K_personal if not specified
        K_personal = args.K_personal if args.K_personal is not None else K

        data = run_ditto(
            feeders,
            test_data,
            model,
            client_optim,
            args.T,  # Total rounds
            M,  # Clients per round
            K,  # Global model local steps
            K_personal,  # Personalized model local steps
            args.B,  # Batch size
            args.ditto_lambda,  # Ditto regularization parameter
            bn_setting=bn_setting,
            noisy_idxs=noisy_idxs,
        )

    save_data(data, fname)
    print("Data saved to: {}".format(fname))


if __name__ == "__main__":
    main()
