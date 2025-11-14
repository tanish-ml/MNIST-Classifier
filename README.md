 # MNIST Classifier (PyTorch)

 A simple PyTorch classifier for the MNIST handwritten digits dataset. This repository contains the training notebook, saved model checkpoint, raw MNIST files, and a new visualization notebook that helps inspect learned convolutional filters and feature maps.

 ## Repository structure

 - `MNIST-classifier.ipynb` — Main training notebook: model, data loading, training loop, evaluation, checkpointing, and a training run (interactive).
 - `conv_visualization.ipynb` — New notebook to visualize convolution kernels and intermediate activations (kernels & feature maps). Use this to inspect learned filters or debug representations.
 - `load_model.ipynb` — Helper notebook for loading saved weights and running inference (check contents for usage examples).
 - `analysis.ipynb` — Optional analysis notebook for plotting or inspecting metrics.
 - `best_mnist_cnn.pth` — Saved PyTorch `state_dict` (best checkpoint from training). Load with `map_location` for portability.
 - `MNIST-Data/` — Local MNIST data (IDX files). torchvision will use this directory when downloading/reading the dataset.

 ## Quick overview

 - Model: `SimpleCNN` — two convolutional layers (1→16→32), pooling, and two fully-connected layers mapping to 10 classes.
 - Training: manual training loop with logging, checkpointing (saves best `state_dict`), and early stopping.
 - Visualization: use `conv_visualization.ipynb` to view kernels and activations for sample images.

 ## Requirements

 Minimal recommended packages (add to `requirements.txt` or install manually):

 ```text
 torch           # use the appropriate install command for your OS/CUDA
 torchvision
 matplotlib
 ```

 Install example (PowerShell, CPU-only example):

 ```powershell
 python -m pip install --upgrade pip
 pip install torch torchvision matplotlib
 ```

 For CUDA-enabled installs, select the wheel/command from https://pytorch.org/get-started/locally/.

 ## How to run

 - Open `MNIST-classifier.ipynb` or `conv_visualization.ipynb` in Jupyter / VS Code and run cells in order.
 - Training behavior and hyperparameters live in the `Config` class inside the notebook (change `BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS`, `PATIENCE`, `MODEL_SAVE_PATH`).
 - Data is read from `./MNIST-Data` — torchvision will not re-download if the files already exist.

 PowerShell quick start from the project root:

 ```powershell
 cd "c:\Users\tanish\Tanish Local\Project Pytorch\MNIST Classifier"
 jupyter lab
 ```

 ## Load the saved model (safe across devices)

 Use `map_location` to avoid CPU/GPU mismatches:

 ```python
 from your_notebook_or_script import SimpleCNN, Config
 import torch

 model = SimpleCNN()
 state = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
 model.load_state_dict(state)
 model.to(Config.DEVICE)
 model.eval()
 ```

 ## Visualization (what's available)

 - `conv_visualization.ipynb` contains utilities to:
   - visualize learned convolution kernels (filters) from `conv1` and `conv2` (kernels are normalized per-filter for display),
   - visualize intermediate feature maps (activations) produced by a chosen layer for a single input image,
   - load `best_mnist_cnn.pth` (if present) to inspect learned filters — otherwise the notebook falls back to random initialization so you can compare.

 ## Notes, issues, and recommended fixes

 1. Evaluation loss averaging: the notebook currently sums batch losses and divides by a batch-derived denominator; to be robust, prefer summing `loss.item() * batch_size` and dividing by the total number of samples (or divide by `len(test_loader)` if you intentionally average by batch). This prevents bias when the final batch is smaller.
 2. Fully-connected bottleneck: the current `fc1 -> fc2` (10→10) design is unusually small and may limit accuracy. Consider `fc1 = Linear(32*7*7, 128); fc2 = Linear(128, 10)` with ReLU after `fc1`.
 3. Reproducibility: set seeds (torch, numpy, random) and optionally set `torch.backends.cudnn.deterministic = True` and `benchmark = False` for reproducible runs (at the cost of some speed).
 4. Validation vs test: the notebooks currently use the test set as the validation signal for early stopping. For a canonical evaluation, split out a validation set from training and reserve the test set for final evaluation.
 5. Windows `DataLoader` workers: in notebooks use `num_workers=0`; if running as a script with workers on Windows, add the `if __name__ == '__main__':` guard.

 ## Tips & next steps I can help with

 - Add a `requirements.txt` or `environment.yml` and pin versions.
 - Apply the fixes above directly to `MNIST-classifier.ipynb` (loss averaging, safer save/load, FC layer change).
 - Add TensorBoard logging or save visualizations (PNGs) from `conv_visualization.ipynb`.
 - Convert the training pipeline into a standalone script with CLI flags and proper Windows guards.

 Tell me which of the above you'd like me to do next and I will update the notebook(s) or add the supporting files.
