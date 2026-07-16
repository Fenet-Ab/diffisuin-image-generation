# DDPM from Scratch with PyTorch

This project implements a Denoising Diffusion Probabilistic Model (DDPM) from scratch using PyTorch, inside a single Jupyter notebook: `Diffusion_Model (3).ipynb`.

## What this project does

- Builds a full DDPM pipeline without using high-level diffusion libraries
- Trains a U-Net model to predict Gaussian noise added to images
- Uses the MNIST dataset as the training data
- Demonstrates both the forward diffusion process and the reverse sampling process
- Visualizes sample images, diffusion stages, training loss, and generated digits

## Key components

- `NoiseScheduler`: computes the forward diffusion noise schedule and adds noise to clean images
- `SinusoidalPositionEmbeddings`: converts timesteps into time embeddings for the model
- `ResidualBlock` and `AttentionBlock`: core building blocks used in the U-Net architecture
- `UNet`: the model that learns to predict the noise given a noisy image and timestep
- training loop: optimizes the model using mean squared error between predicted and actual noise
- sampling function: generates new images by reversing the diffusion process

## Dataset

- Uses the MNIST training set
- Images are resized to 28x28 and normalized to the range [-1, 1]
- Model is configured for grayscale digits (`IMAGE_CHANNELS = 1`)

## Notebook structure

1. Install and import required libraries
2. Set training and diffusion configuration
3. Load and visualize MNIST dataset
4. Define beta schedule and diffusion math
5. Implement forward diffusion and visualize noisy images
6. Build the U-Net model with time embeddings, residual blocks, and attention
7. Train the model on noisy MNIST images
8. Generate new samples using the reverse diffusion process
9. Plot training loss and final generated digits

## How to use

1. Open `Diffusion_Model (3).ipynb` in Jupyter or VS Code
2. Run the notebook cells in order
3. The notebook installs required libraries automatically using `pip`
4. Training is performed inside the notebook, and sample generation is shown at the end

## Requirements

The notebook installs the following packages at runtime:

- `torch`
- `torchvision`
- `torchaudio`
- `tqdm`
- `numpy`
- `matplotlib`
- `Pillow`
- `torchmetrics`
- `tensorboard` (optional)

## Notes

- The notebook is meant for learning and experimenting with DDPM concepts
- It is not optimized for production use or large-scale image generation
- The current configuration is tailored for MNIST and may need adjustment for larger datasets
