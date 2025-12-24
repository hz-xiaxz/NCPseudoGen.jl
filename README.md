# NCPseudoGen.jl

**Norm-Conserving Pseudopotential Generation in Julia**

`NCPseudoGen.jl` is a Julia package for generating norm-conserving pseudopotentials (NCPP) using the Troullier-Martins scheme within the Local Density Approximation (LDA). It leverages Julia's scientific computing ecosystem for high-accuracy numerical integration and automatic differentiation.

## Features

*   **All-Electron SCF**: Self-consistent field calculations for atoms using the Numerov method on a shifted exponential grid.
*   **Troullier-Martins Scheme**: Generation of smooth, norm-conserving pseudo-wavefunctions with optimized polynomial coefficients.
*   **Kleinman-Bylander Projectors**: Construction of separable non-local potentials for efficient plane-wave calculations.
*   **High Precision**:
    *   **FiniteDifferences.jl**: Optimized stencils for grid derivatives.
    *   **ForwardDiff.jl**: Machine-precision derivatives for norm-conservation enforcement.
    *   **TaylorSeries.jl**: Exact polynomial derivatives for matching conditions.
*   **LDA Exchange-Correlation**: Perdew-Zunger (1981) parametrization.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/hz-xiaxz/NCPseudoGen.jl")
```

## Usage

### Generating a Pseudopotential

The main entry point is `generate_ncpp`. Here is an example for Aluminum (Z=13):

```julia
using NCPseudoGen

# 1. Define Radial Grid
# R_p=0.005, r_0=1e-6, r_max=15.0, N=6000
grid = ShiftedExpGrid(0.005, 1e-6, 15.0, 6000)

# 2. Define Electron Configuration
# Format: (n, l, occupancy)
# Aluminum: [Ne] 3s² 3p¹
config = [
    (1, 0, 2.0), # 1s
    (2, 0, 2.0), # 2s
    (2, 1, 6.0), # 2p
    (3, 0, 2.0), # 3s (Valence)
    (3, 1, 1.0)  # 3p (Valence)
]

# 3. Generate Pseudopotential
# Auto-detects core/valence partition and selects cutoff radii if not provided
pp, ae_results = generate_ncpp(grid, 13.0, config)

# 4. Access Results
println("Local Potential (l=$(pp.l_local)):")
# ...
```

### Visualization

See `scripts/plot_ncpp.jl` for examples of how to visualize the generated potentials and wavefunctions.

## Theory

This package solves the radial Schrödinger equation:
$$-\frac{1}{2} \frac{d^2u}{dr^2} + \left[V_{\text{eff}}(r) - E\right] u(r) = 0$$

It uses a **two-way Numerov integration** with matching at the classical turning point to ensure numerical stability for both deep core states and extended valence states.

## References

1.  Troullier, N. & Martins, J. L. (1991). *Phys. Rev. B* 43, 1993.
2.  Kleinman, L. & Bylander, D. M. (1982). *Phys. Rev. Lett.* 48, 1425.
3.  Perdew, J. P. & Zunger, A. (1981). *Phys. Rev. B* 23, 5048.
