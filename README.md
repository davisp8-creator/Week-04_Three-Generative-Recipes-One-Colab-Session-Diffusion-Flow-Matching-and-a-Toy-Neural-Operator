# Week_04_Three-Generative-Recipes-One-Colab-Session-Diffusion-Flow-Matching-and-a-Toy-Neural-Operator
Diffusion, Flow Matching, and a Toy Neural Operator

🌊 Halden Marine Systems: Generative & Operator Learning Pilot Scout

Welcome to the proof-of-concept repository for modern generative and operator-learning methods applied to offshore platform wave loading.

📋 The Scenario

As a Simulation Engineer at Halden Marine Systems, the goal of this project is to scout the viability of advanced ML methods for wave loading simulations. Directed by Kwame Boateng, this repository serves as a strict, zero-budget proof-of-concept.

The Catch: Before securing funding for GPU cluster time, we must prove these methods can work conceptually on a single free GPU (e.g., a standard Google Colab instance).

🎯 Project Objectives

This repository contains a single comprehensive Colab notebook/session tackling three distinct toy models:

Denoising Diffusion Probabilistic Model (DDPM)

A DDPM-style model trained on a custom NumPy-generated 2D point distribution (spirals, moons, or checkerboard) or a small public image set.

Flow-Matching / Rectified-Flow Model

Trained on the exact same distribution as the DDPM for a strict apples-to-apples comparison.

1D Neural Operator (FNO or DeepONet)

Trained on input-solution pairs generated from a custom, simple finite-difference Heat or Burgers equation solver.

💥 The Failure Log

To ensure robust deployment and understand edge cases, this project requires the deliberate induction, diagnosis, and repair of at least two of the following training failures:

Mismatched noise schedules

Exploding learning rates

Resolution mismatches between operator training and evaluation grids

All diagnoses, evidence, and repairs are documented in the FAILURE_LOG.md (or detailed in the notebook).

📊 Deliverable: Decision Walkthrough for Leadership

The final output includes a structured decision brief for Kwame Boateng answering the ultimate question: Which recipe warrants a full-scale pilot?

The decision is justified using the following measured baselines:

Sample Quality

Wall-clock time

Peak VRAM usage

Sampling Latency

Relative L2 Error

The brief also outlines exactly what assumptions and scaling bottlenecks must be validated before moving to the Halden GPU cluster.
