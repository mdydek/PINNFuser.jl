# PINNFuser

PINNFuser is a research-oriented Julia library for enhancing ordinary differential equation (ODE) models using Physics-Informed Neural Networks (PINNs).

The project was developed as part of a bachelor’s thesis and focuses on combining physics-based models with data-driven neural network corrections, with particular emphasis on cardiovascular system modeling.

---

## Overview

Physics-Informed Neural Networks (PINNs) allow incorporating known physical laws directly into the training process of neural networks.  
PINNFuser provides a simple interface for applying PINNs to ODE-based models, enabling improved accuracy when only limited measurement data are available.

The library is designed primarily for lumped-parameter cardiovascular models but can be applied to general ODE systems.

---

## Installation

This library is not available via Julia package registry.

Clone the repository and make it available in your Julia environment manually:

```bash
git clone https://github.com/mdydek/PINNFuser.jl
```

## Example Result

The figure below shows an example result obtained using a one-chamber cardiovascular model enhanced with a Physics-Informed Neural Network. The ground truth data were generated using a significantly more complex four-chamber cardiovascular model. The PINN correction allows the simplified ODE model to achieve improved accuracy while maintaining its simplicity.

<img width="1667" height="1112" alt="final_plot_2-1" src="https://github.com/user-attachments/assets/9f68107b-8636-432b-ab8a-4d0cd0b17a38" />

The full implementation of this example, including model definition, training, and extrapolation, is available in [one chamber model example](examples/OneChamberModelCVS/).
