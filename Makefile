.PHONY: help visualize inspect clean install list-datasets

# Default target
help:
	@echo "Available targets:"
	@echo "  make list-datasets - List all available datasets (12 total)"
	@echo "  make visualize     - Generate visualization of first N frames from dataset"
	@echo "  make inspect       - Inspect dataset structure and feature keys"
	@echo "  make install       - Install dependencies"
	@echo "  make clean         - Remove generated image files"

# Run visualization
visualize:
	@PYTHONPATH=src python -c "from oxe_viz.visualize_images import visualize_first_n; visualize_first_n('taco_play', 'train', n=8)"

# List available datasets
list-datasets:
	@PYTHONPATH=src python -c "from oxe_viz.list_datasets import list_available_datasets; datasets = list_available_datasets(); print(f'\nFound {len(datasets)} dataset(s):\n'); [print(f'  {i}. {d}') for i, d in enumerate(datasets, 1)]; print()"

# Run inspection
inspect:
	@PYTHONPATH=src python -c "from oxe_viz.inspect_example import inspect_one; inspect_one('taco_play', 'train')"

# Install dependencies
install:
	pip install -r requirements.txt

# Clean generated files
clean:
	rm -f *.png

