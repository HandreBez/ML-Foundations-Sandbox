# ML Foundations Sandbox 2026

This repository serves as my active proving ground and daily scratchpad for building the mathematical and practical foundations of Machine Learning and Computer Vision. Rather than relying on high-level APIs, the focus here is on raw algorithm implementation, vectorization, and mathematical rigor.

## Structure

* **`01_NumPy_Drills/`** : Transitioning from standard loops to vectorized operations. Focuses on matrix multiplication, broadcasting, and foundational linear algebra operations necessary for building neural networks.
* **`02_PyTorch_Scratch/`** : From-scratch implementations of fundamental models (e.g., Linear Regression, multi-layer perceptrons on MNIST) to solidify the underlying mathematics and mechanics of gradient descent before scaling up.

## Environment & Reproducibility

This repository uses a Conda environment to maintain dependency synchronization across different workstations. 

To recreate the environment:
```bash
conda env create -f environment.yml
conda activate ml-foundations
