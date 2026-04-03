# Clustering Measurement Scripts

This repository contains scripts for running clustering measurements.

## Script organization

* Scripts whose names contain `validation` are intended for validation and testing. These may be more experimental and less polished.
* Production-ready scripts typically start with `desipipe`.

## Getting started

A good entry point is the `options` dictionary, especially after the call to `fill_fiducial_options`.
This dictionary gathers all configuration options available for the clustering measurements.

From there, the best next step is to inspect the docstrings of the functions that compute each statistic, e.g. `spectrum2_tools.py:compute_mesh2_spectrum(...)` for `mesh2_spectrum`, `correlation2_tools.py:compute_particle2_correlation(...)` for `particle2_correlation`

These docstrings describe the expected inputs, available options, and behavior of each measurement routine.
