"""OXE Visualization - Tools for visualizing and evaluating Open X-Embodiment datasets."""

from .data_loader import gcs_pattern, load_raw
from .dataset_schemas import (
    DATASET_SCHEMAS,
    DatasetSchema,
    auto_detect_schema,
    build_feature_description,
    create_trajectory_parser,
    get_action_key,
    get_image_key,
    get_schema,
)
from .evaluate_baseline import evaluate_all_datasets_baseline, evaluate_baseline
from .evaluate_models import (
    EvaluationMetrics,
    evaluate_all_datasets,
    evaluate_model,
    load_model,
    predict_action,
    print_metrics_summary,
)
from .list_datasets import list_available_datasets
from .visualize_images import (
    find_image_key,
    visualize_all_datasets,
    visualize_first_n,
)

__all__ = [
    # Data loading
    "gcs_pattern",
    "load_raw",
    "list_available_datasets",
    # Schema utilities
    "DatasetSchema",
    "DATASET_SCHEMAS",
    "get_schema",
    "auto_detect_schema",
    "build_feature_description",
    "create_trajectory_parser",
    "get_action_key",
    "get_image_key",
    # Evaluation
    "EvaluationMetrics",
    "evaluate_model",
    "evaluate_all_datasets",
    "evaluate_baseline",
    "evaluate_all_datasets_baseline",
    "load_model",
    "predict_action",
    "print_metrics_summary",
    # Visualization
    "find_image_key",
    "visualize_first_n",
    "visualize_all_datasets",
]
