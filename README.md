# AuraScan AI: Quantum-Optical Pathogen Detector

A highly advanced computer vision and biophotonics simulation engine built to calculate hand sterility indices using mathematical physics models.

## 🔬 Scientific Core Architecture
The system models a dual-channel sensory processing pipeline:
1. **Spatial Frequency Domain Imaging (SFDI) Simulation**: Utilizing 2D Fast Fourier Transforms (`np.fft.fft2`) to isolate high spatial frequency scattering coefficients typical of microscopic cell walls.
2. **Temporal Micro-Flow Metrics**: Simulating sub-micron biological flicker noise using Gaussian high-pass color-space differentials.

## 🛠️ Tech Stack & Optimization
- **Core Processing Language**: Python 3.14+
- **Deep Learning Layer**: PyTorch Core Tensor Grid mapping
- **Computer Vision Graphics**: OpenCV Engine with interactive HUD
- **Persistent Data Layers**: Automated HTML Medical-Grade Report Engine & CSV Time-Series Spreadsheet Logging

## 📈 Metric System
The pipeline applies a non-linear exponential decay function to derive the sterility rating based on cluster surface area density:
$$\text{Sterility Index} = 100 \times e^{-35 \times \text{Contamination Ratio}}$$
