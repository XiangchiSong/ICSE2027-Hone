import argparse
from sys import argv


def any_in_list(x, y):
    """
    Args:
        - x: (iterable)
        - y: (iterable)

    Returns:
        True if any items in x are in y.
    """
    return any(x_i in y for x_i in x)


def parse_args():
    """
    Details for the experiment to run are passed via the command line. Some
    experiment settings require specific arguments to be passed (e.g. the
    different FL algorithms require different hyperparameters).

    Returns:
        argparse.Namespace of parsed arguments.
    """
    # common arguments for all algorithms
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dset",
        required=True,
        choices=[
            "mnist",
            "cifar10",
            "cifar100",
            "tinyimagenet",
            "mind",
            "gtsrb",
            "urbansound8k",
        ],
        help="Federated dataset",
    )
    parser.add_argument(
        "-test_fold",
        type=int,
        default=None,
        help="For UrbanSound8K: which fold to use as test set (1-10). If not specified, uses fold 10",
    )
    parser.add_argument(
        "-iid_label",
        type=float,
        default=0.5,
        help="Controls the label heterogeneity (0.0-1.0). "
        "0.0 = minimum heterogeneity (uniform class distribution), "
        "1.0 = maximum heterogeneity (each client has few classes). "
        "Value will be clamped to [0.0, 1.0] range.",
    )
    parser.add_argument(
        "-iid_data",
        type=float,
        default=0.0,
        help="Controls the data quantity heterogeneity (0.0-1.0). "
        "0.0 = balanced data distribution among clients, "
        "1.0 = maximum imbalance in data quantities. "
        "Value will be clamped to [0.0, 1.0] range.",
    )
    parser.add_argument(
        "-alg",
        required=True,
        help="Optimiser",
        choices=[
            "fedavg",
            "pfedme",
            "perfedavg",
            "scaffold",
            "fedprox",
            "hone",
            "fedsampling",
            "clusteredsampling",
            "page",
            "fedala",
            "ditto",
        ],
    )
    parser.add_argument(
        "-C", required=True, type=float, help="Fraction of clients selected per round"
    )
    parser.add_argument("-B", required=True, type=int, help="Client batch size")
    parser.add_argument("-T", required=True, type=int, help="Total rounds")
    parser.add_argument("-E", required=True, type=int, help="Client num epochs")
    parser.add_argument(
        "-W", required=True, type=int, help="Total workers to split data across"
    )
    parser.add_argument("-seed", required=True, type=int, help="Random seed")
    parser.add_argument("-lr", required=True, type=float, help="Client learning rate")
    parser.add_argument(
        "-noisy_frac", required=True, type=float, help="Fraction of noisy clients"
    )

    # specific arguments for different FL & P2P algorithms
    if any_in_list(
        [
            "fedavg",
            "scaffold",
            "fedprox",
            "clusteredsampling",
            "page",
            "fedala",
            "ditto",
        ],
        argv,
    ):
        parser.add_argument(
            "-bn_private",
            choices=["usyb", "us", "yb", "none"],
            required=True,
            help="Patch parameters to keep private",
        )

    if any_in_list(["scaffold", "fedprox"], argv):
        parser.add_argument(
            "-weight_decay",
            required=True,
            type=float,
            help="Weight decay of SCAFFOLD and FedProx",
        )

    if "perfedavg" in argv:
        parser.add_argument(
            "-perfedavg_beta",
            required=True,
            type=float,
            help="PerFedAvg beta parameter",
        )

    if "pfedme" in argv:
        parser.add_argument(
            "-pfedme_lambda", required=True, type=float, help="pFedMe lambda parameter"
        )
        parser.add_argument(
            "-pfedme_beta", required=True, type=float, help="pFedMe beta parameter"
        )

    if "fedprox" in argv:
        parser.add_argument(
            "-fedprox_mu",
            required=True,
            type=float,
            help="The Proximal Regularization Parameter mu."
            "The strength of the proximal regularization term. The larger the value, the closer the model update is to the global model.",
        )

    if "hone" in argv:
        # Component 1: Game Theory Parameters
        # T_warmup is now automatically set to T//10 in main.py
        # Component 2: Sampling Parameters
        # Component 3: Aggregation Parameters
        # The parameters interface for Hone is temporarily hidden.

    if "fedsampling" in argv:
        parser.add_argument(
            "-fedsampling_alpha",
            type=float,
            default=0.5,
            help="Privacy parameter alpha for FedSampling (0.0-1.0). "
            "Controls the probability of true response vs random response. "
            "Higher values mean more true responses (less privacy).",
        )
        parser.add_argument(
            "-fedsampling_m_param",
            type=int,
            default=20,
            help="Upper bound M for random response in FedSampling. "
            "Random responses will be uniformly distributed in [1, M).",
        )

    if "page" in argv:
        # PAGE general parameters
        parser.add_argument(
            "-page_warmup", type=int, default=50, help="DRL warmup rounds (default: 35)"
        )
        parser.add_argument(
            "-page_reward_alpha",
            type=float,
            default=0.5,
            help="Balance factor between generalization and personalization in server reward (default: 0.5)",
        )

        # Client agent parameters
        parser.add_argument(
            "-page_c_hidden1",
            type=int,
            default=20,
            help="Client agent first hidden layer size (default: 40)",
        )
        parser.add_argument(
            "-page_c_hidden2",
            type=int,
            default=15,
            help="Client agent second hidden layer size (default: 30)",
        )
        parser.add_argument(
            "-page_c_actor_lr",
            type=float,
            default=5e-5,
            help="Client agent actor learning rate (default: 1e-4)",
        )
        parser.add_argument(
            "-page_c_critic_lr",
            type=float,
            default=5e-4,
            help="Client agent critic learning rate (default: 1e-3)",
        )

        # Server agent parameters
        parser.add_argument(
            "-page_s_hidden1",
            type=int,
            default=20,
            help="Server agent first hidden layer size (default: 40)",
        )
        parser.add_argument(
            "-page_s_hidden2",
            type=int,
            default=15,
            help="Server agent second hidden layer size (default: 30)",
        )
        parser.add_argument(
            "-page_s_actor_lr",
            type=float,
            default=5e-5,
            help="Server agent actor learning rate (default: 1e-4)",
        )
        parser.add_argument(
            "-page_s_critic_lr",
            type=float,
            default=5e-4,
            help="Server agent critic learning rate (default: 1e-3)",
        )

    if "fedala" in argv:
        # FedALA specific parameters
        parser.add_argument(
            "-fedala_eta",
            required=True,
            type=float,
            default=1.0,
            help="Learning rate for ALA weight optimization (eta)",
        )
        parser.add_argument(
            "-fedala_rand_percent",
            type=int,
            default=10,
            help="Percentage of local data sampled for ALA weight learning",
        )
        parser.add_argument(
            "-fedala_layer_idx",
            type=int,
            default=0,
            help="Layer index for adaptive aggregation (0 for all layers)",
        )
        parser.add_argument(
            "-fedala_threshold",
            type=float,
            default=0.1,
            help="Convergence threshold for ALA optimization loop",
        )
        parser.add_argument(
            "-fedala_num_pre_loss",
            type=int,
            default=10,
            help="Number of recent losses for convergence check",
        )

    if "ditto" in argv:
        # Ditto specific parameters
        parser.add_argument(
            "-ditto_lambda",
            required=True,
            type=float,
            help="Ditto regularization parameter lambda for personalized model",
        )
        parser.add_argument(
            "-K_personal",
            type=int,
            default=None,
            help="Number of local training steps for personalized model. "
            "If not specified, uses same value as K (global model steps)",
        )

    args = parser.parse_args()

    # Set fixed parameters that don't need command line input
    # All algorithms use GPU device
    args.device = "gpu"

    # Hone algorithm specific fixed parameters
    if args.alg == "hone":
        args.bn_private = "none"
        args.c_d = 200.0
        args.epsilon_d = 10.0
        args.c_g = 10.0
        args.c_l = 20.0
        args.epsilon_contrib = 50.0
        args.delta_coalition = 0.2
        args.gamma1 = 0.9
        args.gamma2 = 0.8

    return args
