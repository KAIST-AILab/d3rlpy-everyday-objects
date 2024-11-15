import argparse

from torch.optim.lr_scheduler import CosineAnnealingLR

import d3rlpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hopper-medium-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)

    reward_scaler = d3rlpy.preprocessing.ReturnBasedRewardScaler(
        multiplier=1000.0
    )

    porel = d3rlpy.algos.PorelConfig(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        batch_size=256,
        weight_temp=3.0,
        max_weight=100.0,
        alpha = 0.1,
        epsilon = -1.0,
        reward_scaler=reward_scaler,
    ).create(device=args.gpu)

    # workaround for learning scheduler
    porel.build_with_dataset(dataset)
    assert porel.impl
    scheduler = CosineAnnealingLR(
        porel.impl._modules.actor_optim,  # pylint: disable=protected-access
        500000,
    )

    def callback(algo: d3rlpy.algos.Porel, epoch: int, total_step: int) -> None:
        scheduler.step()

    porel.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=1000,
        save_interval=10,
        callback=callback,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"Porel_{args.dataset}_{args.seed}",
    )


if __name__ == "__main__":
    main()
