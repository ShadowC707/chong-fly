"""Manual pretraining entry point; importing this module never starts training."""
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--meta', required=True, help='Explicit candidate metadata path')
    parser.add_argument('--dataset', required=True, help='Explicit reflex dataset path')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args(argv)
    from optimizer.pretrain import pretrain_policy
    from simulation.policy import ChongFlyMSPPolicy
    policy = ChongFlyMSPPolicy.from_meta(args.meta)
    return pretrain_policy(policy, dataset_path=args.dataset, epochs=args.epochs)


if __name__ == '__main__':
    main()
