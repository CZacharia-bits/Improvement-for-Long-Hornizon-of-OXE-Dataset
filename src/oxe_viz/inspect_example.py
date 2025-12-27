"""Dataset inspection utilities."""
import tensorflow as tf
from .data_loader import load_raw


def inspect_one(dataset_name: str = "taco_play", split: str = "train") -> None:
    """
    Inspect the structure of a dataset example.
    
    Args:
        dataset_name: Name of the dataset to inspect
        split: Dataset split (default: "train")
    """
    ds = load_raw(dataset_name, split)
    
    for raw in ds.take(1):
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        keys = sorted(ex.features.feature.keys())
        
        print(f"\nDataset: {dataset_name} ({split})")
        print(f"{'='*60}")
        print(f"Total keys: {len(keys)}\n")
        
        print("All feature keys:")
        for key in keys:
            print(f"  • {key}")
        
        print("\nImage-related keys:")
        image_keys = [k for k in keys if "image" in k.lower() or "rgb" in k.lower()]
        if image_keys:
            for key in image_keys:
                print(f"  ✓ {key}")
        else:
            print("  (none found)")
        print()
        break
