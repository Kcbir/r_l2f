# Spectral Annihilation for State-Space Memory

**Linear Algebra · Sequence Modeling · State-Space Models**

---

This repository contains the research implementation of a novel memory update mechanism for State-Space Models (SSMs), built around exact spectral erasure of geometrically conflicting memory directions.

The core idea: rather than letting memory decay passively through exponential forgetting, this work derives a closed-form O(D) update that surgically removes conflicting directions from a fixed-rank memory manifold, using the **Sherman-Morrison identity** applied to a **Generalized Rayleigh Quotient**.

## What This Does

Standard continuous-time SSMs suffer from two compounding failure modes:

- **Eigenvalue drift** as the state matrix evolves under prolonged sequence exposure
- **Persistent noise accumulation** from exponential decay, which smooths rather than erases irrelevant memory structure

This work addresses both by deriving the exact spectral annihilation of conflicting directions, eliminating costly matrix inversions entirely. The result is a memory update that is algebraically exact, computationally lean, and geometrically principled.

## Key Contributions

- Architected an experimental **O(D) memory update** for SSMs via the Sherman-Morrison rank-1 correction on a Generalized Rayleigh Quotient framework
- Derived and implemented **exact spectral erasure** of geometrically conflicting memory directions, with no approximation and no inversion
- Demonstrated elimination of the eigenvalue drift and noise accumulation problems inherent in classical exponential decay models

## Repository Structure

```
main.ipynb            # Core experiments and analysis
construct_kg.py       # Knowledge graph construction
process_dataset.py    # Dataset preprocessing pipeline
dataset/              # Training data
Docs/                 # Supporting documentation
```

## Further Details

Full writeup, derivations, and extended analysis are available at **[kabir.codes](https://kabir.codes)**.

---

*Research by Kabir*