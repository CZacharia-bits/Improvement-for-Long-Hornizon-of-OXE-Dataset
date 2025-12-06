import tensorflow as tf
from .data_loader import load_raw

def inspect_one(dataset_name: str = "taco_play", split: str = "train") -> None:
    ds = load_raw(dataset_name, split)
    for raw in ds.take(1):
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        keys = list(ex.features.feature.keys())
        print("All keys:")
        print(keys)
        print("\nKeys that look image-like:")
        for k in keys:
            if "image" in k.lower() or "rgb" in k.lower():
                print("  ", k)
        break

if __name__ == "__main__":
    inspect_one("taco_play", "train")
