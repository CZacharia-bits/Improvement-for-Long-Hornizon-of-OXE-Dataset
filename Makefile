.PHONY: help visualize visualize-all inspect clean install list-datasets

# Default target
help:
	@echo "OXE Dataset Visualization - Available Commands:"
	@echo ""
	@echo "  make visualize-all  - Visualize all 12 datasets (recommended)"
	@echo "  make visualize      - Visualize taco_play dataset only"
	@echo "  make list-datasets  - List all available datasets"
	@echo "  make inspect        - Inspect dataset structure"
	@echo "  make install        - Install dependencies"
	@echo "  make clean          - Remove generated PNG files"
	@echo ""

# Visualize all datasets
visualize-all:
	@echo "Starting visualization of all datasets..."
	@PYTHONPATH=src python -c "from oxe_viz.visualize_images import visualize_all_datasets; visualize_all_datasets('train', n=8)"

# Visualize single dataset
visualize:
	@echo "Visualizing taco_play dataset..."
	@PYTHONPATH=src python -c "from oxe_viz.visualize_images import visualize_first_n; visualize_first_n('taco_play', 'train', n=8)"

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

# Clean generated files
clean:
	@echo "Cleaning generated PNG files..."
	@rm -f *_first_frames.png
	@echo "Done!"
