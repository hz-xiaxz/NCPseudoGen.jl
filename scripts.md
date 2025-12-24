# Scripts

This directory contains utility scripts for `NCPseudoGen.jl`.

## `plot_ncpp.jl`

This script demonstrates the full workflow of generating and visualizing a norm-conserving pseudopotential for Aluminum.

### Usage

```bash
julia --project=. scripts/plot_ncpp.jl
```

### Output

The script generates the following plots in the current directory:

1.  **`wavefunction_comparison.png`**: Comparison of All-Electron (AE) and Pseudo (PS) wavefunctions.
2.  **`pseudopotentials.png`**: The generated ionic and local pseudopotentials.
3.  **`density_comparison.png`**: Comparison of charge densities ($|u(r)|^2$).
4.  **`kb_projectors.png`**: Kleinman-Bylander projectors for non-local channels.
5.  **`logderiv_matching.png`**: Log-derivative matching at the cutoff radius.

### Key Steps in the Script

1.  **Grid Setup**: Initializes a `ShiftedExpGrid`.
2.  **Configuration**: Defines the electron configuration for Al (Z=13).
3.  **Generation**: Calls `generate_ncpp` to perform the SCF calculation and generate the pseudopotential.
4.  **Visualization**: Uses `CairoMakie` to create publication-quality plots of the results.
