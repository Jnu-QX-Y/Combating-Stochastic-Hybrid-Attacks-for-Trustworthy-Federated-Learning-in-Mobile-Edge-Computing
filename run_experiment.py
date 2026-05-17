import subprocess
import sys
import os

def run_experiment(dataset, model, defense, attack_ratio=0.2, iid='non-iid', num_rounds=100, num_clients=50):

    cmd = [
        sys.executable, 'main.py',
        '--dataset', dataset,
        '--model', model,
        '--defense', defense,
        '--iid', iid,
        '--num_rounds', str(num_rounds),
        '--num_clients', str(num_clients),
        '--attack_ratio', str(attack_ratio),
        '--device', 'cuda',
        '--exp_name', f'{dataset}_{model}_{defense}_{iid}_{attack_ratio}',
        '--log_path', './logs'
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running experiment: {result.stderr}")
        return False

    print("Experiment completed successfully!")
    return True



def run_quick_test(dataset, model, attack_ratio, iid):
    print("Running quick test experiment...")
    return run_experiment(
        dataset=dataset,
        model=model,
        defense='pomdpfl',
        attack_ratio=attack_ratio,
        num_rounds=100,
        num_clients=50,
        iid=iid
    )

def main():
    os.makedirs('./logs', exist_ok=True)
    run_quick_test('CIFAR10', 'resnet18', 0.2, 'iid')




if __name__ == '__main__':
    main()