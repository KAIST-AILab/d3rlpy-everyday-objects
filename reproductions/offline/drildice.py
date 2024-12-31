import argparse

import d3rlpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hopper-expert-v2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)

    drildice = d3rlpy.algos.DrilDICEConfig(
        actor_learning_rate=3e-4,
        nu_learning_rate=3e-4,
        f_divergence_type="SoftTV",
        batch_size=512,
        gamma=0.99,
        alpha=0.001,
    ).create(device=args.gpu)

    drildice.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=1000,
        save_interval=10,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"DrilDICE_{args.dataset}_{args.seed}",
    )


if __name__ == "__main__":
    main()
