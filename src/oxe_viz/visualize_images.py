import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # headless VM: save to file
import matplotlib.pyplot as plt

from .data_loader import load_raw

# Use the static camera; change to "steps/observation/rgb_gripper" if you want gripper view instead.
IMAGE_FEATURE_KEY = "steps/observation/rgb_static"

def parse_example(serialized):
    # Image feature is a sequence of encoded frames, so use VarLenFeature(tf.string)
    feature_description = {
        IMAGE_FEATURE_KEY: tf.io.VarLenFeature(tf.string),
    }
    parsed = tf.io.parse_single_example(serialized, feature_description)
    img_bytes_seq = tf.sparse.to_dense(parsed[IMAGE_FEATURE_KEY])  # shape [T]
    return img_bytes_seq

def visualize_first_n(dataset_name: str = "taco_play",
                      split: str = "train",
                      n: int = 8,
                      out_path: str = "taco_play_first_frames.png") -> None:
    ds = load_raw(dataset_name, split)
    ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)

    # Take the first trajectory only
    for img_bytes_seq in ds.take(1):
        img_bytes_seq = img_bytes_seq.numpy()          # array of length T, each entry is bytes
        T = len(img_bytes_seq)
        num = min(n, T)

        fig, axes = plt.subplots(1, num, figsize=(3 * num, 3))
        if num == 1:
            axes = [axes]

        for i in range(num):
            img = tf.io.decode_image(img_bytes_seq[i], channels=3).numpy()
            ax = axes[i]
            ax.imshow(img)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(out_path)
        print(f"Saved {out_path}")
        break
