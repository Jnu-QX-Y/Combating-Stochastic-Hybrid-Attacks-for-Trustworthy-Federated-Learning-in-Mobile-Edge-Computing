import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='POMDP-FL')

    # Dataset and model parameters
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=['MNIST', 'FMNIST', 'CIFAR10'],
                        help='Dataset to use')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'mobilenetv2'],
                        help='Model architecture')
    parser.add_argument('--iid', type=str, default='non-iid',
                        choices=['iid', 'non-iid'],
                        help='Data distribution type: iid or non-iid')

    # Federated learning parameters
    parser.add_argument('--num_clients', type=int, default=50,
                        help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=200,
                        help='Number of FL rounds')
    parser.add_argument('--local_epochs', type=int, default=3,
                        help='Number of local epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight Decay')
    parser.add_argument('--test_batch_size', type=int, default=1024,
                        help='Batch size for testing')

    # Attack parameters
    parser.add_argument('--attack_ratio', type=float, default=0.2,
                        choices=[0, 0.2, 0.4, 0.6],
                        help='Proportion of abnormal clients')
    parser.add_argument('--poisoning_ratio', type=float, default=0.5,
                        help='Ratio of poisoning vs free-rider in abnormal clients')
    parser.add_argument('--attack_prob_min', type=float, default=0.9,
                        help='Minimum attack probability')
    parser.add_argument('--attack_prob_max', type=float, default=1.0,
                        help='Maximum attack probability')

    # Defense method
    parser.add_argument('--defense', type=str, default='pomdpfl',
                        choices=['fedavg', 'krum', 'median', 'foolsgold',
                                 'wefdefense', 'mudhog', 'dmfedmf', 'pomdpfl'],
                        help='Defense method to use')

    # POMDP-FL specific parameters
    parser.add_argument('--ae_retrain_interval', type=int, default=5,
                        help='Autoencoder retraining interval (rounds)')
    parser.add_argument('--ae_incremental_epochs', type=int, default=10,
                        help='Number of epochs for incremental autoencoder training')

    # System parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42, #默认42 41表现好
                        help='Random seed')
    parser.add_argument('--log_path', type=str, default='./logs',
                        help='Path to save logs')
    parser.add_argument('--exp_name', type=str, default='pomdpfl_exp',
                        help='Experiment name')
    parser.add_argument('--run_id', type=int, default=1,
                        help='Run ID for multiple experiments')

    # POMDP-FL hyperparameters
    parser.add_argument('--min_samples', type=int, default=5,
                        help='Minimum samples for DBSCAN clustering')
    parser.add_argument('--threshold_multiplier', type=float, default=1,
                        help='Threshold multiplier for anomaly detection')
    parser.add_argument('--max_threshold_alpha', type=float, default=1.5,
                        help='Maximum threshold alpha for dynamic adjustment')
    parser.add_argument('--threshold_adaptation_rate', type=float, default=0.1,
                        help='Threshold adaptation rate')

    return parser.parse_args()