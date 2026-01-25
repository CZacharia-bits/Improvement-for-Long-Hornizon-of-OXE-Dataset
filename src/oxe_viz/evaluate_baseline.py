"""Simple baseline evaluation - computes dataset statistics without requiring a model."""

import json
from typing import Optional

import numpy as np
import tensorflow as tf

from .data_loader import load_raw
from .dataset_schemas import create_trajectory_parser, get_action_key, get_schema
from .evaluate_models import EvaluationMetrics
from .list_datasets import list_available_datasets


def evaluate_baseline(
    dataset_name: str,
    split: str = "val",
    max_trajectories: Optional[int] = 10,
) -> EvaluationMetrics:
    """
    Compute baseline statistics for a dataset without a model.

    This provides:
    - Action statistics (mean, std, range)
    - Reward statistics
    - Dataset size information

    Args:
        dataset_name: Name of the OXE dataset
        split: Dataset split to use
        max_trajectories: Maximum trajectories to process

    Returns:
        EvaluationMetrics with baseline statistics
    """
    print(f"\n{'=' * 60}")
    print(f"Computing baseline statistics for {dataset_name} ({split})")
    print(f"{'=' * 60}\n")

    # Get schema and create parser
    print(f"Detecting schema for {dataset_name}...")
    try:
        schema = get_schema(dataset_name, split)
        print(f"  Image: {schema.image_primary}")
        print(f"  Action: {schema.action_key or schema.action_world_vector}")
    except Exception as e:
        print(f"✗ Failed to detect schema: {e}")
        raise

    # Load dataset
    print(f"Loading dataset: {dataset_name}...")
    try:
        ds = load_raw(dataset_name, split)
        parser = create_trajectory_parser(schema)
        ds = ds.map(parser, num_parallel_calls=tf.data.AUTOTUNE)
        print("✓ Dataset loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        raise

    # Collect statistics
    all_actions = []
    all_rewards = []
    num_trajectories = 0
    num_steps = 0

    print("\nProcessing trajectories...")
    for traj_idx, trajectory in enumerate(ds):
        if max_trajectories and traj_idx >= max_trajectories:
            break

        try:
            # Extract actions using schema
            action_key = get_action_key(schema, trajectory)
            if action_key:
                actions = trajectory[action_key].numpy()
                if len(actions.shape) >= 1 and len(actions) > 0:
                    all_actions.append(actions)
                    num_steps += len(actions)

            # Extract rewards using schema
            if schema.reward_key and schema.reward_key in trajectory:
                rewards = trajectory[schema.reward_key].numpy()
                if len(rewards) > 0:
                    all_rewards.extend(rewards.flatten().tolist())

            num_trajectories += 1

            if (traj_idx + 1) % 10 == 0:
                print(f"  Processed {traj_idx + 1} trajectories...")

        except Exception as e:
            print(f"  Warning: Skipping trajectory {traj_idx}: {e}")
            continue

    if num_trajectories == 0:
        raise ValueError(f"No valid trajectories found in {dataset_name}")

    # Compute baseline metrics
    print("\nComputing statistics...")

    if all_actions:
        all_actions_array = np.concatenate(all_actions, axis=0)
        # Use mean action as "prediction" for baseline
        mean_action = np.mean(all_actions_array, axis=0)

        # Compute MSE/MAE/RMSE against mean (baseline)
        mse = np.mean((all_actions_array - mean_action) ** 2)
        mae = np.mean(np.abs(all_actions_array - mean_action))
        rmse = np.sqrt(mse)
    else:
        mse = mae = rmse = 0.0

    # Reward statistics
    reward_mean = np.mean(all_rewards) if all_rewards else None
    reward_std = np.std(all_rewards) if all_rewards else None

    metrics = EvaluationMetrics(
        dataset_name=dataset_name,
        model_name="baseline_mean",
        num_trajectories=num_trajectories,
        num_steps=num_steps,
        action_mse=float(mse),
        action_mae=float(mae),
        action_rmse=float(rmse),
        success_rate=None,
        reward_mean=reward_mean,
        reward_std=reward_std,
    )

    print("✓ Baseline statistics computed")
    print(f"\n  Trajectories: {num_trajectories}")
    print(f"  Steps: {num_steps}")
    print(f"  Action MSE (vs mean): {mse:.6f}")
    print(f"  Action MAE (vs mean): {mae:.6f}")
    print(f"  Action RMSE (vs mean): {rmse:.6f}")
    if reward_mean is not None:
        print(f"  Reward mean: {reward_mean:.4f}")
        print(f"  Reward std: {reward_std:.4f}")

    return metrics


def evaluate_all_datasets_baseline(
    split: str = "val",
    max_trajectories_per_dataset: Optional[int] = 10,
    output_file: Optional[str] = "evaluation_results_baseline.json",
) -> list[EvaluationMetrics]:
    """
    Compute baseline statistics for all datasets.

    Args:
        split: Dataset split to use
        max_trajectories_per_dataset: Max trajectories per dataset
        output_file: JSON file to save results

    Returns:
        List of EvaluationMetrics for each dataset
    """
    datasets = list_available_datasets()
    print(f"\n{'=' * 60}")
    print(f"Computing baseline statistics for {len(datasets)} datasets")
    print(f"{'=' * 60}\n")

    all_metrics = []
    successful = []
    failed = []

    for dataset_name in datasets:
        try:
            metrics = evaluate_baseline(
                dataset_name=dataset_name,
                split=split,
                max_trajectories=max_trajectories_per_dataset,
            )
            all_metrics.append(metrics)
            successful.append(dataset_name)
        except Exception as e:
            failed.append((dataset_name, str(e)))
            print(f"\n✗ {dataset_name}: {e}\n")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {len(successful)} successful, {len(failed)} failed")
    print(f"{'=' * 60}")

    if successful:
        print(f"\n✓ Successfully processed ({len(successful)}):")
        for ds in successful:
            print(f"  • {ds}")

    if failed:
        print(f"\n✗ Failed ({len(failed)}):")
        for ds, error in failed:
            print(f"  • {ds}: {error[:60]}...")

    # Save results
    if output_file:
        results = {
            "successful": [m.to_dict() for m in all_metrics],
            "failed": [{"dataset": ds, "error": error} for ds, error in failed],
            "summary": {
                "total_datasets": len(datasets),
                "successful_count": len(successful),
                "failed_count": len(failed),
                "model_name": "baseline_mean",
                "split": split,
            },
        }
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_file}")

    return all_metrics
