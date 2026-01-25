"""Evaluation utilities for SOTA models on OXE datasets."""
import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
from pathlib import Path

from .data_loader import load_raw
from .list_datasets import list_available_datasets


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    dataset_name: str
    model_name: str
    num_trajectories: int
    num_steps: int
    action_mse: float
    action_mae: float
    action_rmse: float
    success_rate: Optional[float] = None
    reward_mean: Optional[float] = None
    reward_std: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "num_trajectories": self.num_trajectories,
            "num_steps": self.num_steps,
            "action_mse": float(self.action_mse),
            "action_mae": float(self.action_mae),
            "action_rmse": float(self.action_rmse),
            "success_rate": float(self.success_rate) if self.success_rate is not None else None,
            "reward_mean": float(self.reward_mean) if self.reward_mean is not None else None,
            "reward_std": float(self.reward_std) if self.reward_std is not None else None,
        }


def parse_trajectory(serialized: tf.Tensor) -> Dict[str, tf.Tensor]:
    """
    Parse a TFRecord example to extract trajectory data.
    
    Args:
        serialized: Serialized TFRecord example
    
    Returns:
        Dictionary with parsed features
    """
    feature_description = {
        # Actions
        "steps/action/actions": tf.io.VarLenFeature(tf.float32),
        "steps/action/rel_actions_gripper": tf.io.VarLenFeature(tf.float32),
        "steps/action/rel_actions_world": tf.io.VarLenFeature(tf.float32),
        "steps/action/terminate_episode": tf.io.VarLenFeature(tf.float32),  # Can be float or int
        
        # Observations
        "steps/observation/rgb_static": tf.io.VarLenFeature(tf.string),
        "steps/observation/rgb_gripper": tf.io.VarLenFeature(tf.string),
        "steps/observation/robot_obs": tf.io.VarLenFeature(tf.float32),
        "steps/observation/natural_language_instruction": tf.io.VarLenFeature(tf.string),
        
        # Episode info
        "steps/is_first": tf.io.VarLenFeature(tf.int64),
        "steps/is_last": tf.io.VarLenFeature(tf.int64),
        "steps/is_terminal": tf.io.VarLenFeature(tf.int64),
        "steps/reward": tf.io.VarLenFeature(tf.float32),
    }
    
    parsed = tf.io.parse_single_example(serialized, feature_description)
    
    # Convert sparse to dense
    result = {}
    for key, value in parsed.items():
        if isinstance(value, tf.SparseTensor):
            result[key] = tf.sparse.to_dense(value)
        else:
            result[key] = value
    
    return result


def load_model(model_name: str, model_path: Optional[str] = None):
    """
    Load a pre-trained model for evaluation.
    
    Supported models:
    - "rt-1": RT-1 (Robotic Transformer 1)
    - "rt-2": RT-2 (Vision-Language-Action)
    - "octo": Octo model
    - "openvla": OpenVLA model
    - "custom": Custom model from path
    
    Args:
        model_name: Name of the model to load
        model_path: Optional path to model weights (for custom models)
    
    Returns:
        Loaded model object
    
    Raises:
        NotImplementedError: If model loading is not yet implemented
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower == "rt-1":
        # RT-1 model loading
        raise NotImplementedError(
            "RT-1 model loading not yet implemented. "
            "To implement: Install transformers library and load from Hugging Face Hub."
        )
    elif model_name_lower == "rt-2":
        # RT-2 model loading
        raise NotImplementedError(
            "RT-2 model loading not yet implemented. "
            "To implement: Install transformers library and load from Hugging Face Hub."
        )
    elif model_name_lower == "octo":
        # Octo model loading from octo-models library
        try:
            import octo
            from octo.model.octo_model import OctoModel
            
            # Load Octo base model (pre-trained on OXE datasets)
            # Model checkpoint is downloaded automatically on first use
            print("  Loading Octo model from checkpoint...")
            print("  Note: First run will download ~2GB model weights")
            
            # Load the base Octo model from HuggingFace
            # According to Octo README, use HuggingFace path format
            model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")
            
            print("  ✓ Octo model loaded successfully")
            return model
            
        except ImportError as e:
            error_msg = (
                "Octo library not installed or missing dependencies.\n"
                "  To install Octo:\n"
                "  1. Clone the repository: git clone https://github.com/octo-models/octo.git\n"
                "  2. cd octo && pip install -e .\n"
                "  3. Install additional dependencies: pip install torch Pillow\n"
                "  4. Install JAX/Flax (if missing): pip install 'jax[cuda12]' flax\n"
                "     (or 'jax[cpu]' for CPU-only)\n"
                "  Note: Octo requires JAX/Flax, which may need to be installed separately.\n"
                "  Alternative: Use a different model (rt-1, rt-2, openvla) if available.\n"
                f"  Original error: {e}"
            )
            raise ImportError(error_msg)
        except Exception as e:
            raise RuntimeError(f"Failed to load Octo model: {e}")
    elif model_name_lower == "openvla":
        # OpenVLA model loading
        raise NotImplementedError(
            "OpenVLA model loading not yet implemented. "
            "To implement: Install openvla library and load from checkpoint."
        )
    elif model_name_lower == "custom" and model_path:
        # Load custom model from path
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            raise ValueError(f"Failed to load custom model from {model_path}: {e}")
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported: rt-1, rt-2, octo, openvla, custom"
        )


def predict_action(
    model: Any,
    observation: Dict[str, tf.Tensor],
    model_name: str
) -> np.ndarray:
    """
    Predict action from observation using the model.
    
    Args:
        model: Loaded model
        observation: Dictionary with observation data
        model_name: Name of the model (for model-specific preprocessing)
    
    Returns:
        Predicted action array
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower == "octo":
        # Octo model prediction
        try:
            import torch
            from PIL import Image
            import io
            
            # Extract images from observation
            # Octo expects images in a specific format
            if "steps/observation/rgb_static" in observation:
                image_bytes = observation["steps/observation/rgb_static"].numpy()
            elif "steps/observation/rgb_gripper" in observation:
                image_bytes = observation["steps/observation/rgb_gripper"].numpy()
            else:
                raise ValueError("No image found in observation")
            
            # Decode first image (Octo uses single image for prediction)
            if len(image_bytes) > 0:
                img = tf.io.decode_image(image_bytes[0], channels=3).numpy()
            else:
                raise ValueError("Empty image sequence")
            
            # Convert to PIL Image and then to tensor format Octo expects
            pil_img = Image.fromarray(img)
            
            # Extract language instruction if available
            instruction = ""
            if "steps/observation/natural_language_instruction" in observation:
                inst_bytes = observation["steps/observation/natural_language_instruction"].numpy()
                if len(inst_bytes) > 0:
                    instruction = inst_bytes[0].decode('utf-8')
            
            # Prepare observation dict for Octo
            # Octo expects: {"image_primary": image, "language_instruction": text}
            obs_dict = {
                "image_primary": pil_img,
            }
            if instruction:
                obs_dict["language_instruction"] = instruction
            
            # Run inference with Octo
            # Octo.predict() returns action predictions
            action = model.predict(obs_dict)
            
            # Convert to numpy array
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            elif isinstance(action, dict):
                # Octo might return dict with action key
                action = action.get("action", list(action.values())[0])
                if isinstance(action, torch.Tensor):
                    action = action.detach().cpu().numpy()
            
            return np.array(action)
            
        except ImportError as e:
            raise ImportError(f"Missing dependencies for Octo prediction: {e}")
        except Exception as e:
            raise RuntimeError(f"Octo prediction failed: {e}")
    
    else:
        raise NotImplementedError(
            f"Action prediction for {model_name} not yet implemented. "
            f"Currently only 'octo' is supported."
        )


def evaluate_model(
    model_name: str,
    dataset_name: str,
    split: str = "val",
    max_trajectories: Optional[int] = None,
    max_steps_per_trajectory: Optional[int] = None,
    model_path: Optional[str] = None,
) -> EvaluationMetrics:
    """
    Evaluate a model on an OXE dataset.
    
    Args:
        model_name: Name of the model to evaluate
        dataset_name: Name of the OXE dataset
        split: Dataset split to use (default: "val")
        max_trajectories: Maximum number of trajectories to evaluate (None = all)
        max_steps_per_trajectory: Maximum steps per trajectory (None = all)
        model_path: Optional path to model weights
    
    Returns:
        EvaluationMetrics object with computed metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on {dataset_name} ({split})")
    print(f"{'='*60}\n")
    
    # Load model
    print(f"Loading model: {model_name}...")
    try:
        model = load_model(model_name, model_path)
        print(f"✓ Model loaded successfully")
    except NotImplementedError as e:
        print(f"✗ {e}")
        raise
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        raise
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}...")
    try:
        ds = load_raw(dataset_name, split)
        ds = ds.map(parse_trajectory, num_parallel_calls=tf.data.AUTOTUNE)
        print(f"✓ Dataset loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        raise
    
    # Collect predictions and ground truth
    all_predicted_actions = []
    all_ground_truth_actions = []
    all_rewards = []
    num_trajectories = 0
    num_steps = 0
    
    print(f"\nRunning evaluation...")
    for traj_idx, trajectory in enumerate(ds):
        if max_trajectories and traj_idx >= max_trajectories:
            break
        
        # Extract ground truth actions
        if "steps/action/actions" in trajectory:
            gt_actions = trajectory["steps/action/actions"].numpy()
        elif "steps/action/rel_actions_world" in trajectory:
            gt_actions = trajectory["steps/action/rel_actions_world"].numpy()
        else:
            print(f"  Warning: No action found in trajectory {traj_idx}, skipping")
            continue
        
        # Extract rewards if available
        if "steps/reward" in trajectory:
            rewards = trajectory["steps/reward"].numpy()
            all_rewards.extend(rewards.flatten().tolist())
        
        # Limit steps per trajectory
        if max_steps_per_trajectory:
            gt_actions = gt_actions[:max_steps_per_trajectory]
        
        # Predict actions using the model
        # Process each step in the trajectory
        predicted_actions_list = []
        
        # Extract observations for each step
        num_steps_in_traj = len(gt_actions)
        for step_idx in range(num_steps_in_traj):
            # Create observation dict for this step
            step_obs = {}
            for key, value in trajectory.items():
                if isinstance(value, tf.Tensor):
                    value_np = value.numpy()
                    if len(value_np.shape) > 0 and step_idx < len(value_np):
                        step_obs[key] = tf.constant([value_np[step_idx]])
                    else:
                        step_obs[key] = value
                else:
                    step_obs[key] = value
            
            try:
                # Predict action for this step
                pred_action = predict_action(model, step_obs, model_name)
                predicted_actions_list.append(pred_action)
            except Exception as e:
                print(f"  Warning: Prediction failed for step {step_idx}: {e}")
                # Use zero action as fallback
                predicted_actions_list.append(np.zeros_like(gt_actions[0]))
        
        # Stack predictions
        if predicted_actions_list:
            predicted_actions = np.stack(predicted_actions_list, axis=0)
            # Ensure shape matches ground truth
            if predicted_actions.shape != gt_actions.shape:
                # Try to reshape or pad
                if len(predicted_actions.shape) == 1 and len(gt_actions.shape) == 2:
                    predicted_actions = predicted_actions.reshape(-1, 1)
                # Pad or truncate to match
                min_len = min(len(predicted_actions), len(gt_actions))
                predicted_actions = predicted_actions[:min_len]
                gt_actions = gt_actions[:min_len]
        else:
            # Fallback: use zeros
            predicted_actions = np.zeros_like(gt_actions)
        
        all_predicted_actions.append(predicted_actions)
        all_ground_truth_actions.append(gt_actions)
        num_trajectories += 1
        num_steps += len(gt_actions)
        
        if (traj_idx + 1) % 10 == 0:
            print(f"  Processed {traj_idx + 1} trajectories...")
    
    if num_trajectories == 0:
        raise ValueError(f"No valid trajectories found in {dataset_name}")
    
    # Compute metrics
    print(f"\nComputing metrics...")
    all_predicted_actions = np.concatenate(all_predicted_actions, axis=0)
    all_ground_truth_actions = np.concatenate(all_ground_truth_actions, axis=0)
    
    # Action prediction metrics
    mse = np.mean((all_predicted_actions - all_ground_truth_actions) ** 2)
    mae = np.mean(np.abs(all_predicted_actions - all_ground_truth_actions))
    rmse = np.sqrt(mse)
    
    # Reward metrics
    reward_mean = np.mean(all_rewards) if all_rewards else None
    reward_std = np.std(all_rewards) if all_rewards else None
    
    # Success rate (if terminal flags available)
    success_rate = None  # Would need to check terminal flags
    
    metrics = EvaluationMetrics(
        dataset_name=dataset_name,
        model_name=model_name,
        num_trajectories=num_trajectories,
        num_steps=num_steps,
        action_mse=float(mse),
        action_mae=float(mae),
        action_rmse=float(rmse),
        success_rate=success_rate,
        reward_mean=reward_mean,
        reward_std=reward_std,
    )
    
    print(f"✓ Evaluation complete")
    return metrics


def evaluate_all_datasets(
    model_name: str,
    split: str = "val",
    max_trajectories_per_dataset: Optional[int] = 10,
    max_steps_per_trajectory: Optional[int] = None,
    model_path: Optional[str] = None,
    output_file: Optional[str] = None,
) -> List[EvaluationMetrics]:
    """
    Evaluate a model on all available OXE datasets.
    
    Args:
        model_name: Name of the model to evaluate
        split: Dataset split to use (default: "val")
        max_trajectories_per_dataset: Max trajectories per dataset (None = all)
        max_steps_per_trajectory: Max steps per trajectory (None = all)
        model_path: Optional path to model weights
        output_file: Optional JSON file to save results
    
    Returns:
        List of EvaluationMetrics for each dataset
    """
    datasets = list_available_datasets()
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on {len(datasets)} datasets")
    print(f"{'='*60}\n")
    
    all_metrics = []
    successful = []
    failed = []
    
    for dataset_name in datasets:
        try:
            metrics = evaluate_model(
                model_name=model_name,
                dataset_name=dataset_name,
                split=split,
                max_trajectories=max_trajectories_per_dataset,
                max_steps_per_trajectory=max_steps_per_trajectory,
                model_path=model_path,
            )
            all_metrics.append(metrics)
            successful.append(dataset_name)
            
            # Print summary
            print(f"\n{dataset_name}:")
            print(f"  Trajectories: {metrics.num_trajectories}")
            print(f"  Steps: {metrics.num_steps}")
            print(f"  Action MSE: {metrics.action_mse:.6f}")
            print(f"  Action MAE: {metrics.action_mae:.6f}")
            print(f"  Action RMSE: {metrics.action_rmse:.6f}")
            
        except Exception as e:
            failed.append((dataset_name, str(e)))
            print(f"\n✗ {dataset_name}: {e}\n")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(successful)} successful, {len(failed)} failed")
    print(f"{'='*60}")
    
    if successful:
        print(f"\n✓ Successfully evaluated ({len(successful)}):")
        for ds in successful:
            print(f"  • {ds}")
    
    if failed:
        print(f"\n✗ Failed ({len(failed)}):")
        for ds, error in failed:
            print(f"  • {ds}: {error[:60]}...")
    
    # Save results (including error information)
    if output_file:
        results = {
            "successful": [m.to_dict() for m in all_metrics],
            "failed": [{"dataset": ds, "error": error} for ds, error in failed],
            "summary": {
                "total_datasets": len(datasets),
                "successful_count": len(successful),
                "failed_count": len(failed),
                "model_name": model_name,
                "split": split
            }
        }
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_file}")
        if len(failed) > 0:
            print(f"  ⚠ Warning: {len(failed)} dataset(s) failed. Check 'failed' section in JSON.")
    
    return all_metrics


def print_metrics_summary(metrics: EvaluationMetrics) -> None:
    """Print a formatted summary of evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {metrics.model_name} on {metrics.dataset_name}")
    print(f"{'='*60}")
    print(f"Trajectories evaluated: {metrics.num_trajectories}")
    print(f"Total steps: {metrics.num_steps}")
    print(f"\nAction Prediction Metrics:")
    print(f"  MSE:  {metrics.action_mse:.6f}")
    print(f"  MAE:  {metrics.action_mae:.6f}")
    print(f"  RMSE: {metrics.action_rmse:.6f}")
    if metrics.reward_mean is not None:
        print(f"\nReward Statistics:")
        print(f"  Mean: {metrics.reward_mean:.4f}")
        print(f"  Std:  {metrics.reward_std:.4f}")
    if metrics.success_rate is not None:
        print(f"\nSuccess Rate: {metrics.success_rate:.2%}")
    print(f"{'='*60}\n")
