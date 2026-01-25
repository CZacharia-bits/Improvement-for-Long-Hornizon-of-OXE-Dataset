.PHONY: help visualize visualize-all inspect clean install list-datasets evaluate evaluate-all evaluate-baseline

# Default target
help:
	@echo "OXE Dataset Visualization & Evaluation - Available Commands:"
	@echo ""
	@echo "Visualization:"
	@echo "  make visualize-all  - Visualize all 12 datasets (recommended)"
	@echo "  make visualize      - Visualize taco_play dataset only"
	@echo ""
	@echo "Evaluation:"
	@echo "  make evaluate         - Evaluate a model on taco_play dataset"
	@echo "  make evaluate-all     - Evaluate a model on all datasets"
	@echo "  make evaluate-baseline - Compute baseline stats (no model needed)"
	@echo ""
	@echo "Utilities:"
	@echo "  make list-datasets  - List all available datasets"
	@echo "  make inspect        - Inspect dataset structure"
	@echo "  make install        - Install dependencies"
	@echo "  make clean          - Remove generated PNG files"
	@echo ""

# Visualize all datasets
visualize-all:
	@echo "Starting visualization of all datasets..."
	@PYTHONPATH=src python -c "from oxe_viz.visualize_images import visualize_all_datasets; visualize_all_datasets('train', n=None)"

# Visualize single dataset
visualize:
	@echo "Visualizing taco_play dataset..."
	@PYTHONPATH=src python -c "from oxe_viz.visualize_images import visualize_first_n; visualize_first_n('taco_play', 'train', n=None)"

# List available datasets
list-datasets:
	@PYTHONPATH=src python -c "from oxe_viz.list_datasets import list_available_datasets; datasets = list_available_datasets(); print(f'\nFound {len(datasets)} dataset(s):\n'); [print(f'  {i}. {d}') for i, d in enumerate(datasets, 1)]; print()"

# Inspect dataset structure
inspect:
	@PYTHONPATH=src python -c "from oxe_viz.inspect_example import inspect_one; inspect_one('taco_play', 'train')"

# Install dependencies
install:
	@echo "Installing dependencies..."
	@pip install -r requirements.txt

# Evaluate model on single dataset
evaluate:
	@echo "Evaluating model on taco_play dataset..."
	@echo "Look for ✓ checkmarks and metrics summary to verify success."
	@PYTHONPATH=src python -c "from oxe_viz.evaluate_models import evaluate_model, print_metrics_summary; metrics = evaluate_model('octo', 'taco_play', 'train', max_trajectories=10); print_metrics_summary(metrics)"

# Evaluate model on all datasets
evaluate-all:
	@echo "Evaluating model on all datasets..."
	@PYTHONPATH=src python -c "from oxe_viz.evaluate_models import evaluate_all_datasets; evaluate_all_datasets('octo', 'train', max_trajectories_per_dataset=5, output_file='evaluation_results.json')"

# Evaluate baseline (no model required)
evaluate-baseline:
	@echo "Computing baseline statistics for all datasets..."
	@PYTHONPATH=src python -c "from oxe_viz.evaluate_baseline import evaluate_all_datasets_baseline; evaluate_all_datasets_baseline('val', max_trajectories_per_dataset=10, output_file='evaluation_results_baseline.json')"

# Clean generated files
clean:
	@echo "Cleaning generated PNG files..."
	@rm -f *_first_frames.png
	@rm -f evaluation_results.json
	@echo "Done!"
