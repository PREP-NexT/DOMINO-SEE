<p align="center">
  <a href="https://hem-w.github.io/dominosee-dev/">
    <img src="https://raw.githubusercontent.com/Hem-W/dominosee-dev/refs/heads/main/docs/source/_static/DOMINO-SEE%20Horizontal.svg" width="550" alt="DOMINO-SEE logo">
  </a>
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Build" src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple"></a>
  <a href="https://github.com/Hem-W/dominosee-dev"><img src="https://img.shields.io/github/contributors/Hem-W/dominosee-dev.svg" alt="GitHub contributors"></a>
  <a href="https://github.com/Hem-W/dominosee-dev"><img src="https://img.shields.io/github/issues/Hem-W/dominosee-dev.svg" alt="GitHub issues"></a>
  <a href="https://twitter.com/PREPNexT_Lab"><img src="https://img.shields.io/twitter/follow/PREPNexT_Lab.svg?label=Follow&style=social" alt="Twitter Follow"></a>
  <a href="https://github.com/Hem-W/dominosee-dev"><img src="https://img.shields.io/github/license/Hem-W/dominosee-dev.svg" alt="License"></a>
  <a href="https://github.com/Hem-W/dominosee-dev"><img src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103" alt="Download"></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> |
  <a href="#why-the-name">Why the Name?</a> |
  <a href="#key-features">Key Features</a> |
  <a href="#getting-started">Getting Started</a> |
  <a href="#contact-us">Contact Us</a> |
  <a href="#disclaimer">Disclaimer</a>
</p>

## Overview

**DOMINO-SEE** (**D**etection Of **M**ulti-layer **IN**terconnected **O**ccurrences for **S**patial **E**xtreme **E**vents) is a data-driven statistical framework for detecting spatially co-occurrences of hydroclimatic extreme events across locations, inspired by complex network science, powered by `xarray` architecture. It's developed by [Hui-Min Wang](https://orcid.org/0000-0002-5878-7542) and [Xiaogang He](https://cde.nus.edu.sg/cee/staff/he-xiaogang/) from the [PREP-NexT](https://github.com/PREP-NexT) Lab at the [National University of Singapore](https://nus.edu.sg/).

This project is licensed under the [GNU General Public License 2.0](https://github.com/PREP-NexT/DOMINO-SEE/blob/main/LICENSE).

## Why the Name?

The name **DOMINO-SEE** represents our approach to detecting and analyzing interconnected occurrences of hydroclimatic extreme events across spatial locations, inspired by the cascade effect of **DOMINO**es falling in a chain reaction. The **SEE** highlights the framework's ability to capture the spatial synchronization and propagation of extreme events, emphasizing the interconnectedness inherent in complex environmental systems.

## Key Features

- **Complex Network Generation**: Fast and memory-efficient functions to build spatial networks from event series among spatial locations and multiple types (layers) of climate extreme events
- **Multi-dimensional Support**: Native support for `xarray` DataArrays to handle multi-dimensional gridded climate data
- **Parallel Processing**: `dask` integration for efficient processing of large-scale climate datasets
- **Blockwise Computation**: Utilities for splitting large spatial datasets into manageable blocks of netCDF datasets (see `dominosee/utils/blocking.py`).

## Getting Started

This section includes a brief tutorial on running your first DOMINO-SEE model.

1. Clone the repo

    ```bash
    git clone https://github.com/Hem-W/dominosee-dev.git
    ```

2. Install the dependencies through conda

    ```bash
    cd dominosee-dev
    conda env create -f environment.yml
    conda activate dominosee
    ```

3. Install the package from source

    ```bash
    pip install -e .
    ```

## Contact Us

If you have any questions, comments, or suggestions that aren't suitable for public discussion in the Issues section, please feel free to contact [Hui-Min Wang](mailto:wanghuimin@u.nus.edu).

Please use the GitHub Issues for public discussions related to bugs, enhancements, or other project-related discussions.

## Disclaimer

The DOMINO-SEE model is an academic project and is not intended to be used as a precise prediction tool for risk assessment and management. The developers will not be held liable for any decisions made based on the use of this model. We recommend applying it in conjunction with expert judgment and other modeling tools in a decision-making context.
