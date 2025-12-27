"""Data loading utilities for OXE datasets from Google Cloud Storage."""
import tensorflow as tf
from .config import OXEDatasetConfig


def gcs_pattern(dataset_name: str, split: str = "train") -> str:
    """
    Construct GCS path pattern for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        split: Dataset split (e.g., "train", "val")
    
    Returns:
        GCS path pattern string
    """
    cfg = OXEDatasetConfig()
    version = "0.1.0" if dataset_name != "language_table" else "0.0.1"
    return (
        f"{cfg.base_path}/{dataset_name}/{version}/"
        f"{dataset_name}-{split}.tfrecord-*"
    )


def load_raw(dataset_name: str, split: str = "train") -> tf.data.Dataset:
    """
    Load raw TFRecord dataset from GCS.
    
    Args:
        dataset_name: Name of the dataset
        split: Dataset split (default: "train")
    
    Returns:
        TensorFlow dataset of raw TFRecord examples
    
    Raises:
        ValueError: If no files found for the pattern
    """
    pattern = gcs_pattern(dataset_name, split)
    files = tf.io.gfile.glob(pattern)
    if not files:
        raise ValueError(f"No TFRecord files found for pattern: {pattern}")
    return tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
