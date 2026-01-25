import tensorflow as tf

from .config import OXEDatasetConfig


def list_available_datasets() -> list[str]:
    """
    List all available datasets in the OXE GCS bucket.

    Returns:
        List of dataset names found in the bucket.
    """
    cfg = OXEDatasetConfig()
    base_path = cfg.base_path

    # List all directories (datasets) in the base path
    try:
        # List all items in the base directory
        items = tf.io.gfile.listdir(base_path)
        datasets = []

        for item in items:
            # Remove trailing slash if present
            item = item.rstrip("/")
            # Check if it's a directory (dataset name)
            item_path = f"{base_path}/{item}"
            if tf.io.gfile.isdir(item_path):
                # Check if it contains version directories
                try:
                    subdirs = tf.io.gfile.listdir(item_path)
                    # If it has version directories, it's likely a dataset
                    if any(subdir.startswith("0.") for subdir in subdirs):
                        datasets.append(item)
                except Exception:
                    pass

        return sorted(datasets)
    except Exception as e:
        print(f"Error listing datasets: {e}")
        print(f"Attempted to access: {base_path}")
        return []


if __name__ == "__main__":
    datasets = list_available_datasets()
    print(f"\nFound {len(datasets)} dataset(s):\n")
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i}. {dataset}")
    print()
