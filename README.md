# MUD-EXP3 and AMUD-EXP3
This repository includes the implementation of our work **"A Modified EXP3 and Its Adaptive Variant in Adversarial Bandits with Multi-User Delayed Feedback"**.

## Introduction
A brief introduction about the files:
-  `mud.py` and  `amud.py`: our proposed methods of MUD-EXP3 and AMUD-EXP3.
- `ducb.py`, `se.py` and `rand.py`: the baseline methods.
- `oracle.py': the oracle method that always chooses the optimal arm.
- `env.py`: the simulation environments including stochastic and adversarial bandits.
- `utils.py`: the generation function of truncated normal distributions for `env.py`.
- `plotter.py`: the plotting methods for demonstrate the experimental results.

## Prerequisites
- Python3
- Numpy
- Matplotlib
- Scipy

## Numerical Examples
One can run the file `main.py` to get the cumulative loss line plot with respect to the rounds.
