"""Dataset schema definitions and auto-detection for OXE datasets.

Different OXE datasets have varying schemas for actions, images, and metadata.
This module provides:
- DatasetSchema: A dataclass defining the mapping from canonical names to raw keys
- DATASET_SCHEMAS: Registry of known dataset schemas
- auto_detect_schema(): Fallback for unknown datasets
- get_schema(): Main entry point for retrieving schemas
- build_feature_description(): Generate TFRecord feature descriptions
- create_trajectory_parser(): Factory for tf.data.Dataset.map() parsers
"""

from dataclasses import dataclass, field
from typing import Optional

import tensorflow as tf

from .data_loader import load_raw


@dataclass
class DatasetSchema:
    """Schema mapping for an OXE dataset.

    Maps canonical feature names to the actual keys used in the TFRecord files.
    """

    # Image keys (canonical -> raw)
    image_primary: Optional[str] = None
    image_wrist: Optional[str] = None

    # Action keys
    action_key: Optional[str] = None  # Combined action tensor
    action_world_vector: Optional[str] = None  # Position delta
    action_rotation: Optional[str] = None  # Rotation delta
    action_gripper: Optional[str] = None  # Gripper action
    action_terminate: Optional[str] = None  # Episode termination

    # Metadata
    language_key: Optional[str] = None
    reward_key: Optional[str] = None
    is_first_key: Optional[str] = None
    is_last_key: Optional[str] = None
    is_terminal_key: Optional[str] = None

    # Robot state
    robot_obs_key: Optional[str] = None

    # Type overrides (key -> tf.dtype)
    dtype_overrides: dict[str, tf.DType] = field(default_factory=dict)

    # Special handling
    uses_zlib_compression: bool = False


# =============================================================================
# DATASET REGISTRY
# =============================================================================

DATASET_SCHEMAS: dict[str, DatasetSchema] = {
    # === FAILING DATASETS (need explicit schemas) ===
    "fractal20220817_data": DatasetSchema(
        image_primary="steps/observation/image",
        action_world_vector="steps/action/world_vector",
        action_rotation="steps/action/rotation_delta",
        action_gripper="steps/action/gripper_closedness_action",
        language_key="steps/observation/natural_language_instruction",
        reward_key="steps/reward",
        is_first_key="steps/is_first",
        is_last_key="steps/is_last",
        is_terminal_key="steps/is_terminal",
        dtype_overrides={
            "steps/is_first": tf.bool,
            "steps/is_last": tf.bool,
            "steps/is_terminal": tf.bool,
        },
    ),
    "jaco_play": DatasetSchema(
        image_primary="steps/observation/image",
        image_wrist="steps/observation/image_wrist",
        action_world_vector="steps/action/world_vector",
        action_gripper="steps/action/gripper_closedness_action",
        language_key="steps/observation/natural_language_instruction",
        reward_key="steps/reward",
        is_first_key="steps/is_first",
        is_last_key="steps/is_last",
        is_terminal_key="steps/is_terminal",
    ),
    "kuka": DatasetSchema(
        image_primary="steps/observation/image",
        action_world_vector="steps/action/world_vector",
        action_rotation="steps/action/rotation_delta",
        action_gripper="steps/action/gripper_closedness_action",
        language_key="steps/observation/natural_language_instruction",
        reward_key="steps/reward",
        is_first_key="steps/is_first",
        is_last_key="steps/is_last",
        is_terminal_key="steps/is_terminal",
        uses_zlib_compression=True,
    ),
    # === WORKING DATASETS (document for completeness) ===
    "taco_play": DatasetSchema(
        image_primary="steps/observation/rgb_static",
        image_wrist="steps/observation/rgb_gripper",
        action_key="steps/action/rel_actions_world",
        action_gripper="steps/action/rel_actions_gripper",
        action_terminate="steps/action/terminate_episode",
        language_key="steps/observation/natural_language_instruction",
        reward_key="steps/reward",
        robot_obs_key="steps/observation/robot_obs",
        is_first_key="steps/is_first",
        is_last_key="steps/is_last",
        is_terminal_key="steps/is_terminal",
    ),
    "taco_extra": DatasetSchema(
        image_primary="steps/observation/rgb_static",
        image_wrist="steps/observation/rgb_gripper",
        action_key="steps/action/rel_actions_world",
        action_gripper="steps/action/rel_actions_gripper",
        action_terminate="steps/action/terminate_episode",
        language_key="steps/observation/natural_language_instruction",
        reward_key="steps/reward",
        robot_obs_key="steps/observation/robot_obs",
        is_first_key="steps/is_first",
        is_last_key="steps/is_last",
        is_terminal_key="steps/is_terminal",
    ),
    "berkeley_autolab_ur5": DatasetSchema(
        image_primary="steps/observation/image",
        image_wrist="steps/observation/hand_image",
        action_world_vector="steps/action/world_vector",
        action_rotation="steps/action/rotation_delta",
        action_gripper="steps/action/gripper_closedness_action",
        language_key="steps/observation/natural_language_instruction",
        reward_key="steps/reward",
        is_first_key="steps/is_first",
        is_last_key="steps/is_last",
        is_terminal_key="steps/is_terminal",
    ),
    "bridge": DatasetSchema(
        image_primary="steps/observation/image_0",
        image_wrist="steps/observation/image_1",
        action_world_vector="steps/action/world_vector",
        action_rotation="steps/action/rotation_delta",
        action_gripper="steps/action/open_gripper",
        language_key="steps/observation/natural_language_instruction",
        reward_key="steps/reward",
        is_first_key="steps/is_first",
        is_last_key="steps/is_last",
        is_terminal_key="steps/is_terminal",
    ),
    "language_table": DatasetSchema(
        image_primary="steps/observation/rgb",
        action_world_vector="steps/action",
        language_key="steps/observation/instruction",
        reward_key="steps/reward",
        is_first_key="steps/is_first",
        is_last_key="steps/is_last",
        is_terminal_key="steps/is_terminal",
    ),
    "bc_z": DatasetSchema(
        image_primary="steps/observation/image",
        action_key="steps/action/future/xyz_residual",
        action_rotation="steps/action/future/axis_angle_residual",
        action_gripper="steps/action/future/target_close",
        language_key="steps/observation/natural_language_instruction",
        reward_key="steps/reward",
        is_first_key="steps/is_first",
        is_last_key="steps/is_last",
        is_terminal_key="steps/is_terminal",
    ),
    "rt_1_x": DatasetSchema(
        image_primary="steps/observation/image",
        action_world_vector="steps/action/world_vector",
        action_rotation="steps/action/rotation_delta",
        action_gripper="steps/action/gripper_closedness_action",
        action_terminate="steps/action/terminate_episode",
        language_key="steps/observation/natural_language_instruction",
        reward_key="steps/reward",
        is_first_key="steps/is_first",
        is_last_key="steps/is_last",
        is_terminal_key="steps/is_terminal",
    ),
    "nyu_door_opening_surprising_effectiveness": DatasetSchema(
        image_primary="steps/observation/image",
        action_world_vector="steps/action/world_vector",
        action_rotation="steps/action/rotation_delta",
        action_gripper="steps/action/gripper_closedness_action",
        language_key="steps/observation/natural_language_instruction",
        reward_key="steps/reward",
        is_first_key="steps/is_first",
        is_last_key="steps/is_last",
        is_terminal_key="steps/is_terminal",
    ),
    "stanford_hydra_dataset_converted_externally_to_rlds": DatasetSchema(
        image_primary="steps/observation/image",
        image_wrist="steps/observation/wrist_image",
        action_world_vector="steps/action/world_vector",
        action_rotation="steps/action/rotation_delta",
        action_gripper="steps/action/gripper_closedness_action",
        language_key="steps/observation/natural_language_instruction",
        reward_key="steps/reward",
        is_first_key="steps/is_first",
        is_last_key="steps/is_last",
        is_terminal_key="steps/is_terminal",
    ),
}


# =============================================================================
# AUTO-DETECTION
# =============================================================================


def auto_detect_schema(dataset_name: str, split: str = "train") -> DatasetSchema:
    """Auto-detect schema by inspecting the first record of a dataset.

    Args:
        dataset_name: Name of the OXE dataset
        split: Dataset split to inspect

    Returns:
        DatasetSchema with detected key mappings

    Raises:
        ValueError: If the dataset is empty
    """
    ds = load_raw(dataset_name, split)

    for raw in ds.take(1):
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        keys = set(ex.features.feature.keys())

        schema = DatasetSchema()

        # Find image keys
        for key in sorted(keys):
            key_lower = key.lower()
            if "image" in key_lower or "rgb" in key_lower:
                if "wrist" in key_lower or "gripper" in key_lower or "hand" in key_lower:
                    if schema.image_wrist is None:
                        schema.image_wrist = key
                elif schema.image_primary is None:
                    schema.image_primary = key

        # Find action keys
        for key in sorted(keys):
            key_lower = key.lower()
            if "action" in key_lower:
                if "world" in key_lower or "xyz" in key_lower:
                    if schema.action_world_vector is None:
                        schema.action_world_vector = key
                elif "rotation" in key_lower or "axis_angle" in key_lower:
                    if schema.action_rotation is None:
                        schema.action_rotation = key
                elif "gripper" in key_lower or "close" in key_lower:
                    if schema.action_gripper is None:
                        schema.action_gripper = key
                elif "terminate" in key_lower:
                    if schema.action_terminate is None:
                        schema.action_terminate = key
                elif schema.action_key is None:
                    # Generic action key (could be combined action tensor)
                    schema.action_key = key

        # Find metadata keys
        for key in sorted(keys):
            key_lower = key.lower()
            if "instruction" in key_lower or "language" in key_lower:
                if schema.language_key is None:
                    schema.language_key = key
            elif key.endswith("reward"):
                if schema.reward_key is None:
                    schema.reward_key = key
            elif key.endswith("is_first"):
                if schema.is_first_key is None:
                    schema.is_first_key = key
            elif key.endswith("is_last"):
                if schema.is_last_key is None:
                    schema.is_last_key = key
            elif key.endswith("is_terminal"):
                if schema.is_terminal_key is None:
                    schema.is_terminal_key = key

        # Find robot observation key
        for key in sorted(keys):
            if "robot_obs" in key.lower():
                if schema.robot_obs_key is None:
                    schema.robot_obs_key = key

        return schema

    raise ValueError(f"Dataset {dataset_name} is empty")


# =============================================================================
# SCHEMA LOOKUP
# =============================================================================


def get_schema(dataset_name: str, split: str = "train") -> DatasetSchema:
    """Get schema from registry or auto-detect.

    Args:
        dataset_name: Name of the OXE dataset
        split: Dataset split (used for auto-detection)

    Returns:
        DatasetSchema for the dataset
    """
    if dataset_name in DATASET_SCHEMAS:
        return DATASET_SCHEMAS[dataset_name]
    return auto_detect_schema(dataset_name, split)


# =============================================================================
# PARSER FACTORY
# =============================================================================


def build_feature_description(schema: DatasetSchema) -> dict[str, tf.io.VarLenFeature]:
    """Build TFRecord feature description from schema.

    Args:
        schema: DatasetSchema defining the key mappings

    Returns:
        Dictionary suitable for tf.io.parse_single_example()
    """
    features: dict[str, tf.io.VarLenFeature] = {}

    def add_feature(key: Optional[str], default_dtype: tf.DType) -> None:
        if key:
            actual_dtype = schema.dtype_overrides.get(key, default_dtype)
            features[key] = tf.io.VarLenFeature(actual_dtype)

    # Images (bytes)
    add_feature(schema.image_primary, tf.string)
    add_feature(schema.image_wrist, tf.string)

    # Actions (float32)
    add_feature(schema.action_key, tf.float32)
    add_feature(schema.action_world_vector, tf.float32)
    add_feature(schema.action_rotation, tf.float32)
    add_feature(schema.action_gripper, tf.float32)
    add_feature(schema.action_terminate, tf.float32)

    # Metadata
    add_feature(schema.language_key, tf.string)
    add_feature(schema.reward_key, tf.float32)

    # Episode markers (int64 by default, can be overridden to bool)
    add_feature(schema.is_first_key, tf.int64)
    add_feature(schema.is_last_key, tf.int64)
    add_feature(schema.is_terminal_key, tf.int64)

    # Robot state
    add_feature(schema.robot_obs_key, tf.float32)

    return features


def create_trajectory_parser(
    schema: DatasetSchema,
) -> tf.types.experimental.ConcreteFunction:
    """Create a parser function for tf.data.Dataset.map().

    Args:
        schema: DatasetSchema defining the key mappings

    Returns:
        A function that parses serialized TFRecord examples
    """
    feature_desc = build_feature_description(schema)

    def parse(serialized: tf.Tensor) -> dict[str, tf.Tensor]:
        parsed = tf.io.parse_single_example(serialized, feature_desc)
        result: dict[str, tf.Tensor] = {}
        for key, value in parsed.items():
            if isinstance(value, tf.SparseTensor):
                result[key] = tf.sparse.to_dense(value)
            else:
                result[key] = value
        return result

    return parse


def get_action_key(schema: DatasetSchema, trajectory: dict[str, tf.Tensor]) -> Optional[str]:
    """Find the best action key to use from a trajectory.

    Tries keys in order of preference:
    1. Combined action tensor (action_key)
    2. World vector (position delta)
    3. Rotation delta
    4. Gripper action

    Args:
        schema: DatasetSchema for the dataset
        trajectory: Parsed trajectory dictionary

    Returns:
        The key to use for actions, or None if no action found
    """
    # Try combined action first
    if schema.action_key and schema.action_key in trajectory:
        return schema.action_key

    # Try world vector
    if schema.action_world_vector and schema.action_world_vector in trajectory:
        return schema.action_world_vector

    # Try rotation
    if schema.action_rotation and schema.action_rotation in trajectory:
        return schema.action_rotation

    # Try gripper
    if schema.action_gripper and schema.action_gripper in trajectory:
        return schema.action_gripper

    return None


def get_image_key(schema: DatasetSchema, trajectory: dict[str, tf.Tensor]) -> Optional[str]:
    """Find the best image key to use from a trajectory.

    Args:
        schema: DatasetSchema for the dataset
        trajectory: Parsed trajectory dictionary

    Returns:
        The key to use for images, or None if no image found
    """
    if schema.image_primary and schema.image_primary in trajectory:
        return schema.image_primary

    if schema.image_wrist and schema.image_wrist in trajectory:
        return schema.image_wrist

    return None
