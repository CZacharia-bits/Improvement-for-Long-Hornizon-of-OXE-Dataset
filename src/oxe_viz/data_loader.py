import tensorflow as tf
from .config import OXEDatasetConfig

def gcs_pattern(dataset_name: str, split: str = "train") -> str:
    cfg = OXEDatasetConfig()
    version = "0.1.0" if dataset_name != "language_table" else "0.0.1"
    return (
        f"{cfg.base_path}/{dataset_name}/{version}/"
        f"{dataset_name}-{split}.tfrecord-*"
    )

def load_raw(dataset_name: str, split: str = "train") -> tf.data.Dataset:
    pattern = gcs_pattern(dataset_name, split)
    files = tf.io.gfile.glob(pattern)
    if not files:
        raise ValueError(f"No TFRecord files found for pattern: {pattern}")
    return tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
