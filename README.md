<p align="center">
  <a href="https://github.com/PREP-NexT/DOMINO-SEE">
    <img src="https://raw.githubusercontent.com/PREP-NexT/DOMINO-SEE/refs/heads/main/docs/source/_static/DOMINO-SEE%20Horizontal.svg" width="550" alt="DOMINO-SEE logo">
  </a>
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Build" src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple"></a>
  <a href="https://github.com/PREP-NexT/DOMINO-SEE"><img src="https://img.shields.io/github/contributors/PREP-NexT/DOMINO-SEE.svg" alt="GitHub contributors"></a>
  <a href="https://github.com/PREP-NexT/DOMINO-SEE"><img src="https://img.shields.io/github/issues/PREP-NexT/DOMINO-SEE.svg" alt="GitHub issues"></a>
  <a href="https://twitter.com/PREPNexT_Lab"><img src="https://img.shields.io/twitter/follow/PREPNexT_Lab.svg?label=Follow&style=social" alt="Twitter Follow"></a>
  <a href="https://github.com/PREP-NexT/DOMINO-SEE"><img src="https://img.shields.io/github/license/PREP-NexT/DOMINO-SEE.svg" alt="License"></a>
  <a href="https://github.com/PREP-NexT/DOMINO-SEE"><img src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103" alt="Download"></a>
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

## Why the Name?

The name **DOMINO-SEE** represents our approach to detecting and analyzing interconnected occurrences of hydroclimatic extreme events across spatial locations, inspired by the cascade effect of **DOMINO**es falling in a chain reaction. The **SEE** highlights the framework's ability to capture the spatial synchronization and propagation of extreme events, emphasizing the interconnectedness inherent in complex environmental systems.

## Key Features

- **Complex Network Generation**: Fast and memory-efficient functions to build spatial networks from event series among spatial locations and multiple types (layers) of climate extreme events
- **Multi-dimensional Support**: Native support for `xarray` DataArrays to handle multi-dimensional gridded climate data
- **Parallel Processing**: `dask` integration for efficient processing of large-scale climate datasets
- **Grid Generation**: Equidistant Fekete grid generation for alternative spatial embedding.
<!-- - **Blockwise Computation**: Utilities for splitting large spatial datasets into manageable blocks of netCDF datasets (see `dominosee/utils/blocking.py`). -->

<!-- ## Development Status

This project is under active development. Current implementation status:

- ✅ **FeketeGrid**: Equidistant grid on a sphere - fully implemented and tested

The grid module is being uploaded as a work-in-progress to facilitate collaborative development. Only `BaseGrid` and `FeketeGrid` are currently recommended for production use. -->

## Getting Started

This section includes a brief tutorial on running your first DOMINO-SEE model.

1. Clone the repo

    ```bash
    git clone https://github.com/PREP-NexT/DOMINO-SEE.git
    ```

2. Install the dependencies through conda

    ```bash
    cd DOMINO-SEE
    conda env create -f environment.yml
    conda activate dominosee
    ```

3. Install the package from source

    ```bash
    pip install -e .
    ```

## Citing DOMINO-SEE

![DOMINO-SEE_logos_QR](https://raw.githubusercontent.com/PREP-NexT/DOMINO-SEE/refs/heads/main/docs/source/_static/images/dominosee_banner_qr_white.svg)

If you use DOMINO-SEE in a scientific publication, we kindly ask that you cite our article published in Nature Water:

<table>
  <tr>
    <td>
      Wang, H.-M., &amp; He, X. (2025). Spatially synchronized structures of global hydroclimatic extremes.
      <em>Nature Water</em>. https://doi.org/10.1038/s44221-025-00520-w
    </td>
  </tr>
</table>

You may also use the following BibTeX entry:

```bibtex
@article{wang_2025,
	title = {Spatially synchronized structures of global hydroclimatic extremes},
	issn = {2731-6084},
	url = {https://www.nature.com/articles/s44221-025-00520-w},
	doi = {10.1038/s44221-025-00520-w},
	urldate = {2025-10-29},
	journal = {Nature Water},
	author = {Wang, Hui-Min and He, Xiaogang},
	month = oct,
	year = {2025},
}
```

## Contact Us

DOMINO-SEE is still under active development by [Hui-Min Wang](mailto:wanghuimin@u.nus.edu) from the [PREP-NexT Lab](https://github.com/PREP-NexT).

- If you're interested in suggesting new features or reporting bugs, please leave us a message on the [***issue tracker***](https://github.com/PREP-NexT/DOMINO-SEE/issues).

- If you have any questions, comments, or suggestions that aren't suitable for public discussion in Issues, please feel free to contact [Hui-Min Wang](mailto:wanghuimin@u.nus.edu).

## Disclaimer

This project is licensed under the [GNU General Public License 3.0](https://github.com/PREP-NexT/DOMINO-SEE/blob/main/LICENSE). The DOMINO-SEE model is an academic project and is not intended to be used as a precise prediction tool for risk assessment and management. The developers will not be held liable for any decisions made based on the use of this model. We recommend applying it in conjunction with expert judgment and other modeling tools in a decision-making context.
