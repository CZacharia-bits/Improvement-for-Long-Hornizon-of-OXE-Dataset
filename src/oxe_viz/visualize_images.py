"""Visualization utilities for OXE datasets."""
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # headless VM: save to file
import matplotlib.pyplot as plt

from .data_loader import load_raw

# Common image feature keys to try (in order of preference)
IMAGE_FEATURE_KEYS = [
    "steps/observation/rgb_static",
    "steps/observation/image",
    "steps/observation/rgb_gripper",
    "steps/observation/hand_image",
]


def find_image_key(dataset_name: str, split: str = "train") -> str:
    """Find the first available image feature key in a dataset."""
    ds = load_raw(dataset_name, split)
    
    # Get first example to inspect
    for raw in ds.take(1):
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        available_keys = set(ex.features.feature.keys())
        
        # Try each key in order of preference
        for key in IMAGE_FEATURE_KEYS:
            if key in available_keys:
                return key
        
        # If none found, look for any key with "image" or "rgb"
        for key in available_keys:
            if "image" in key.lower() or "rgb" in key.lower():
                return key
        
        raise ValueError(
            f"No image feature found in {dataset_name}. "
            f"Available keys: {sorted(available_keys)}"
        )
    
    raise ValueError(f"Dataset {dataset_name} is empty")


def parse_example(serialized, image_key: str):
    """Parse a TFRecord example to extract image sequence."""
    feature_description = {
        image_key: tf.io.VarLenFeature(tf.string),
    }
    parsed = tf.io.parse_single_example(serialized, feature_description)
    img_bytes_seq = tf.sparse.to_dense(parsed[image_key])
    return img_bytes_seq


def visualize_first_n(
    dataset_name: str = "taco_play",
    split: str = "train",
    n: int = 8,
    out_path: str = None,
    image_key: str = None,
) -> None:
    """
    Visualize the first N frames from a dataset.
    
    Args:
        dataset_name: Name of the dataset
        split: Dataset split (default: "train")
        n: Number of frames to visualize (default: 8)
        out_path: Output file path (default: "{dataset_name}_first_frames.png")
        image_key: Image feature key to use (auto-detected if None)
    """
    if out_path is None:
        out_path = f"{dataset_name}_first_frames.png"
    
    # Auto-detect image key if not provided
    if image_key is None:
        image_key = find_image_key(dataset_name, split)
    
    # Load and parse dataset
    ds = load_raw(dataset_name, split)
    ds = ds.map(lambda x: parse_example(x, image_key), num_parallel_calls=tf.data.AUTOTUNE)
    
    # Extract first trajectory
    for img_bytes_seq in ds.take(1):
        img_bytes_seq = img_bytes_seq.numpy()
        T = len(img_bytes_seq)
        
        if T == 0:
            raise ValueError(f"No frames found in first trajectory for {dataset_name}")
        
        num = min(n, T)
        
        # Create figure
        fig, axes = plt.subplots(1, num, figsize=(3 * num, 3))
        if num == 1:
            axes = [axes]
        
        # Display frames
        for i in range(num):
            img = tf.io.decode_image(img_bytes_seq[i], channels=3).numpy()
            axes[i].imshow(img)
            axes[i].axis("off")
        
        plt.tight_layout()
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Saved {out_path} ({num} frames from {image_key})")
        return
    
    raise ValueError(f"No trajectories found in {dataset_name}")


def visualize_all_datasets(split: str = "train", n: int = 8) -> None:
    """
    Visualize all available datasets from the OXE collection.
    
    Args:
        split: Dataset split to use (default: "train")
        n: Number of frames to visualize per dataset (default: 8)
    """
    from .list_datasets import list_available_datasets
    
    datasets = list_available_datasets()
    print(f"\n{'='*60}")
    print(f"Visualizing {len(datasets)} datasets...")
    print(f"{'='*60}\n")
    
    successful = []
    failed = []
    
    for dataset_name in datasets:
        try:
            out_path = f"{dataset_name}_first_frames.png"
            visualize_first_n(dataset_name, split, n, out_path)
            successful.append(dataset_name)
        except Exception as e:
            failed.append((dataset_name, str(e)))
            print(f"✗ {dataset_name}: {e}\n")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(successful)} successful, {len(failed)} failed")
    print(f"{'='*60}")
    
    if successful:
        print(f"\n✓ Successfully visualized ({len(successful)}):")
        for ds in successful:
            print(f"  • {ds}_first_frames.png")
    
    if failed:
        print(f"\n✗ Failed ({len(failed)}):")
        for ds, error in failed:
            print(f"  • {ds}: {error[:60]}...")
    
    print()
