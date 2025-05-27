<p align="center">
  <a href="https://github.com/PREP-NexT/DOMINO-SEE">
    <img src="https://raw.githubusercontent.com/Hem-W/dominosee-dev/refs/heads/main/docs/source/_static/DOMINO-SEE%20Horizontal.svg" width="550" alt="DOMINO-SEE logo">
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
  <a href="#how-to-use">How to Use?</a> |
  <a href="#prerequisites">Prerequisites</a> |
  <a href="#run-scripts">Run Scripts</a> |
  <a href="#contact-us">Contact Us</a>
</p>

## Overview

**DOMINO-SEE** (**D**etection Of **M**ulti-layer **IN**terconnected **O**ccurrences for **S**patial **E**xtreme **E**vents) is a data-driven statistical framework for detecting spatially co-occurrences of hydroclimatic extreme events across locations, inspired by complex network science, powered by `xarray` architecture. It's developed by [Hui-Min Wang](https://orcid.org/0000-0002-5878-7542) and [Xiaogang He](https://cde.nus.edu.sg/cee/staff/he-xiaogang/) from the [PREP-NexT](https://github.com/PREP-NexT) Lab at the [National University of Singapore](https://nus.edu.sg/).

This project is licensed under the [GNU General Public License 2.0](https://github.com/PREP-NexT/DOMINO-SEE/blob/main/LICENSE).

## Why the Name?

The name **DOMINO-SEE** represents our approach to detecting and analyzing interconnected occurrences of hydroclimatic extreme events across spatial locations, inspired by the cascade effect of **DOMINO**es falling in a chain reaction. The **SEE** highlights the framework's ability to capture the spatial synchronization and propagation of extreme events, emphasizing the interconnectedness inherent in complex environmental systems.

## How to Use?

This repository stores the code for reproducing the results in the paper. A more general repository for constructing event-based climate networks using `xarray` can be found [here](https://github.com/Hem-W/DOMINOSEE-dev).

## Prerequisites

This section includes a brief tutorial on running your first DOMINO-SEE model.

1. Clone the repo

```bash
git clone https://github.com/PREP-NexT/DOMINO-SEE.git
```

2. Install the dependencies

```bash
cd DOMINO-SEE
conda env create -f environment.yml
conda activate DOMINO-SEE
```

3. Prepare the data

+ SPI1 data: placed at `0data/SPI1_monthly_0.250deg_1950_2016.nc`.
+ A boolean array of valid points with monthly precipitation data > 100mm: placed at `0data/prcpfkt_validpoint_annual_100.npy`.
+ SST data: placed at `0data/sst.mnmean.nc`.

## Run scripts

*The scripts should be run in the following order:*

#### 1. `0spi_events.py`

Loads SPI (Standardized Precipitation Index) data and identifies drought/flood events based on predefined thresholds (e.g., SPI ≤ -1.5 for drought, SPI ≥ 1.5 for flood). Calculates event timing, bursts, and durations, and saves results in `1events/`.

#### 2. `1eca_rate.py`

Implements Event Coincidence Analysis (ECA) to detect synchronous events across different locations, which are saved in `2eca/`.

#### 3. `1eca_null.py`

Generates a null model distribution for statistical significance testing of ECA, which are saved in `2eca/null/`.

#### 4. `1eca_sig.py`

Compares actual Event Coincidence Analysis results against the null model to identify statistically significant coincidence rates, which are saved in `3link/`.

#### 5. `2link_network.py`

Constructs a network from significant ECA links and calculates the great circle distances between connected locations. Separates links into teleconnections (≥2500km) and short-range connections (<2500km). 

#### 6. `3bipartite_network.py`

Analyzes teleconnection networks between different geographical regions using kernel density estimation. Identifies significant spatial link densities and linked regional bundles between regions.

*The plotting functions are also available as follows:*

* **plot_event_number.ipynb** - Plots the annual mean number of drought/flood events across the global domain.

* **plot_link_distance_compare.py** - Compares the distance distributions of links across different network types (drought-drought, pluvial-pluvial, drought-pluvial).

* **plot_global_degree_tele.py** - Produces maps of teleconnection network links (≥ 2500km) showing the spatial distribution of long-distance connections.

* **plot_global_degree_short.py** - Generates maps of short-distance network connections (< 2500km) showing the spatial distribution of link densities for drought and pluvial networks.

* **plot_crop_region.py** - Creates visualizations of specific geographical regions for detailed analysis of climate patterns.

* **plot_bipartite_network.py** - Creates visualizations of teleconnection networks between different geographical regions, showing linked bundles and their spatial densities.

* **plot_pairwise_network_density.py** - Visualizes the density of connections between specific regional pairs, highlighting the strength of climate teleconnections.

* **plot_SST.py** - Visualizes sea surface temperature (SST) anomalies during synchronized climate events between different regions, with region boxes highlighting areas of interest.

## Contact Us

If you have any questions, comments, or suggestions that aren't suitable for public discussion in the Issues section, please feel free to contact [Hui-Min Wang](mailto:wanghuimin@u.nus.edu).

## Disclaimer

The DOMINO-SEE model is an academic project and is not intended to be used as a precise prediction tool for commercial or policy-making purposes without expert oversight. The developers will not be held liable for any decisions made based on the use of this model. We recommend applying it in conjunction with expert judgment and other modeling tools in a decision-making context.

