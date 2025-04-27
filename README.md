# tma4850-acoustic-wave-sim

**TMA4850: Acoustic Wave Simulation**\
Course project for TMA4850: Experts in Teamwork – Mathematics in Applications.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [OpenFOAM Simulation](#openfoam-simulation)
  - [Python Finite-Difference Solver](#python-finite-difference-solver)
- [Results and Post-Processing](#results-and-post-processing)
- [Authors](#authors)
- [License](#license)

## Overview

This project provides two implementations for simulating acoustic wave propagation:

1. **OpenFOAM (finite-volume)** – Finite Volume implementation with mesh algorithm and plate placement algorithm.
2. **Python FDTD (finite-difference)** – Python finite difference implementation with frequency dependent boundary condition. 

## Repository Structure

```
tma4850-acoustic-wave-sim
├── OpenFOAM/
│   ├── Allrun.sh
│   ├── master.py
│   ├── integrated_pressure_new.py
│   ├── post_process.py
│   └── setManufactured.py
└── PythonFD/
    ├── scheme_freq_dep.py
    └── post_processing.py
```

## Prerequisites

- **OpenFOAM** (confirmed functioning for version 2312 and 2406)
- **Python 3.8+**

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/tma4850-acoustic-wave-sim.git
   cd tma4850-acoustic-wave-sim
   ```
2. For Python Finite Difference implementation also get https://github.com/SebastianL18/Vector_Fitting_for_python

## Usage

### OpenFOAM Simulation

1. Navigate to the `OpenFOAM` directory:
   ```bash
   cd OpenFOAM
   ```
2. Make the all-run script executable and run:
   ```bash
   chmod +x Allrun.sh
   ./Allrun.sh
   ```
   This script sets up the mesh and runs the simulation
3. For plate algorithm run `master.py`
   ```bash
   python3 master.py
   ```

### Python Finite-Difference Solver

1. Navigate to `PythonFD`:
   ```bash
   cd PythonFD
   ```
2. Run the solver script:
   ```bash
   python3 scheme_freq_dep.py
   ```
3. Post-process results:
   ```bash
   python3 post_processing.py
   ```
4. Generated plots will be saved in an `images/` directory.

## Results and Post-Processing

- Compute metrics reverberation time (RT60) and speech clarity (C50).

## Authors

- **Experts in Teamwork** group 6, TMA4850 – Mathematics in Applications.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

