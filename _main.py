import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from datetime import datetime
import json

# Import custom modules
from dataloader import Dataset_CIFAR10, Dataset_MNIST, Dataset_FMNIST
from models.init import get_model
from clients import Client
from clients_attackers import FreeRiderClient, PoisoningClient
from server import Server


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_path, exp_name, run_id):
    """Setup logging configuration"""
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, f'{exp_name}_{run_id}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_dataset(dataset_name, data_path='./data'):
    """Load and return train/test datasets"""
    if dataset_name == 'CIFAR10':
        train_dataset = Dataset_CIFAR10()
        train_dataset.load(path=os.path.join(data_path, 'CIFAR10'), train=True)
        test_dataset = Dataset_CIFAR10()
        test_dataset.load(path=os.path.join(data_path, 'CIFAR10'), train=False)
    elif dataset_name == 'MNIST':
        train_dataset = Dataset_MNIST()
        train_dataset.load(path=os.path.join(data_path, 'MNIST'), train=True)
        test_dataset = Dataset_MNIST()
        test_dataset.load(path=os.path.join(data_path, 'MNIST'), train=False)
    elif dataset_name == 'FMNIST':
        train_dataset = Dataset_FMNIST()
        train_dataset.load(path=os.path.join(data_path, 'FMNIST'), train=True)
        test_dataset = Dataset_FMNIST()
        test_dataset.load(path=os.path.join(data_path, 'FMNIST'), train=False)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_dataset, test_dataset


def create_clients(args, client_datasets, device):
    """Create client instances including normal and malicious clients"""
    clients = []

    # Calculate number of malicious clients
    num_abnormal = int(args.num_clients * args.attack_ratio)  # 50* 0 0.2 0.4 0.6
    num_poisoning = int(num_abnormal * args.poisoning_ratio)  # num_abnormal*0.5
    num_freerider = num_abnormal - num_poisoning  #

    logging.info(f"Creating {args.num_clients} clients: "
                 f"{args.num_clients - num_abnormal} normal, "
                 f"{num_poisoning} poisoning, {num_freerider} free-rider")

    # Create client indices and shuffle
    client_indices = list(range(args.num_clients))
    random.shuffle(client_indices)

    # Assign roles
    poisoning_indices = set(client_indices[:num_poisoning])
    freerider_indices = set(client_indices[num_poisoning:num_abnormal])

    for i in range(args.num_clients):
        # Create data loader for this client
        train_loader = DataLoader(
            client_datasets[i],
            batch_size=args.batch_size,
            shuffle=True
        )

        # Create model and optimizer for this client
        client_model = get_model(args.model, args.dataset)
        optimizer = optim.SGD(client_model.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)

        # Generate attack probability for this client
        attack_prob = random.uniform(args.attack_prob_min, args.attack_prob_max)

        if i in poisoning_indices:
            # Create poisoning client
            if args.dataset == 'CIFAR10':
                source_label, target_label = 1, 9  # Cat -> Dog 3 5
            elif args.dataset == 'FMNIST':
                source_label, target_label = 5, 7  # Sandal -> Sneaker
            else:  # MNIST
                source_label, target_label = 1, 9

            client = PoisoningClient(
                cid=i, model=client_model, dataLoader=train_loader,
                optimizer=optimizer, device=device,
                inner_epochs=args.local_epochs,
                source_label=source_label, target_label=target_label,
                attack_prob=attack_prob
            )
        elif i in freerider_indices:
            # Create free-rider client
            noise_std = random.uniform(0.005, 0.01)
            # noise_std = random.uniform(0.08, 0.1)
            client = FreeRiderClient(
                cid=i, model=client_model, dataLoader=train_loader,
                optimizer=optimizer, device=device,
                inner_epochs=args.local_epochs,
                mean=0, std=noise_std,
                attack_prob=attack_prob
            )
        else:
            # Create normal client
            client = Client(
                cid=i, model=client_model, dataLoader=train_loader,
                optimizer=optimizer, device=device,
                inner_epochs=args.local_epochs
            )

        clients.append(client)

    return clients


def main(args):
    """Main function to run the federated learning experiment"""
    # Setup
    set_seed(args.seed)
    setup_logging(args.log_path, args.exp_name, args.run_id)

    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load datasets
    logging.info(f"Loading {args.dataset} dataset...")
    train_dataset, test_dataset = load_dataset(args.dataset)

    # 获取攻击参数用于创建不平衡测试集
    if args.dataset == 'CIFAR10':
        source_label, target_label = 1, 9  # Cat -> Dog
    elif args.dataset == 'FMNIST':
        source_label, target_label = 5, 7  # Sandal -> Sneaker
    else:  # MNIST
        source_label, target_label = 1, 9

    # 创建不平衡测试集（仅在有攻击时）
    if args.attack_ratio > 0:
        logging.info(f"Creating imbalanced test set for attack evaluation...")
        logging.info(f"Attack: {source_label} -> {target_label}")

        # 检查是否有相关参数，如果没有则使用默认值
        source_test_count = getattr(args, 'source_test_count', 5000)
        target_test_count = getattr(args, 'target_test_count', 1000)
        other_test_count = getattr(args, 'other_test_count', 500)

        test_dataset = test_dataset.create_imbalanced_test_set(
            source_label=source_label,
            target_label=target_label,
            source_count=source_test_count,  # 源类样本数
            target_count=target_test_count,  # 目标类样本数
            other_count=other_test_count  # 其他类样本数
        )
    else:
        logging.info("No attack scenario - using balanced test set")

    # Split data among clients
    is_iid = (args.iid == 'iid')
    logging.info(f"Splitting data among {args.num_clients} clients (IID: {args.iid})...")
    client_datasets = train_dataset.split(args.num_clients, iid=is_iid)

    # Create test data loader
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    # Create global model
    global_model = get_model(args.model, args.dataset)
    logging.info(f"Created {args.model} model for {args.dataset}")

    # Create clients
    clients = create_clients(args, client_datasets, device)

    # Create server
    server = Server(global_model, test_loader, device=device)
    server.set_log_path(args.log_path, args.exp_name, args.run_id)

    # Attach clients to server
    for client in clients:
        server.attach(client)

    # Set aggregation rule with args
    server.set_AR(args.defense, args)

    # Log experiment configuration
    logging.info("=== Experiment Configuration ===")
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")
    logging.info("================================")

    # Run federated learning
    logging.info("Starting federated learning...")

    for round_num in range(args.num_rounds):
        logging.info(
            f"\n================================= Round {round_num + 1}/{args.num_rounds} ================================= ")

        # Server distributes global model to clients
        server.distribute()

        # All clients participate in each round (you can modify this for client sampling)
        participating_clients = list(range(len(clients)))

        # Clients train and server aggregates
        server.train(participating_clients)

        # Test global model every few rounds
        test_loss, accuracy = server.test()
        logging.info(f"Round {round_num + 1} - Test Accuracy: {accuracy:.2f}%")

    # Final test
    logging.info("\n=== Final Results ===")
    final_test_loss, final_accuracy = server.test()
    logging.info(f"Final Test Accuracy: {final_accuracy:.2f}%")

    # Save final results
    results = {
        'final_accuracy': final_accuracy,
        'final_test_loss': final_test_loss,
        'experiment_config': vars(args)
    }

    results_file = os.path.join(args.log_path, f'final_results_{args.exp_name}_{args.run_id}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info("Experiment completed successfully!")