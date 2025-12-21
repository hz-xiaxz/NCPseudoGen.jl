# Norm-Conserving Pseudopotential (NCPP) Implementation Plan

## Overview

Generate norm-conserving pseudopotentials from all-electron (AE) atomic calculations using the Troullier-Martins (TM) scheme.

## Theory Background

### Norm Conservation Conditions (Hamann et al. 1979)

For each angular momentum channel l:
1. **Eigenvalue match**: ε_ps = ε_AE
2. **Wavefunction match**: ψ_ps(r) = ψ_AE(r) for r > r_c
3. **Norm conservation**: ∫₀^r_c |ψ_ps|² dr = ∫₀^r_c |ψ_AE|² dr
4. **Log-derivative match**: (d/dr)ln(ψ_ps)|_{r_c} = (d/dr)ln(ψ_AE)|_{r_c}

Condition 4 follows from conditions 2-3 via the identity:
```
d/dε [(d/dr)ln(ψ)] = -2 ∫₀^r |ψ|² dr / (r² ψ²)
```

### Troullier-Martins Scheme

Inside r_c, the pseudo-wavefunction has form:
```
ψ_ps(r) = r^(l+1) exp(p(r))  for r < r_c
```
where p(r) is a polynomial:
```
p(r) = c₀ + c₂r² + c₄r⁴ + c₆r⁶ + c₈r⁸ + c₁₀r¹⁰ + c₁₂r¹²
```

The 7 coefficients are determined by:
1. Continuity of ψ at r_c
2. Continuity of ψ' at r_c
3. Continuity of ψ'' at r_c
4. Continuity of ψ''' at r_c
5. Continuity of ψ'''' at r_c
6. Norm conservation
7. Zero curvature at origin: p''(0) = 0 → c₂ determined

### Pseudopotential Inversion

From the pseudo-wavefunction, invert the radial Schrödinger equation:
```
V_ps,scr(r) = ε - l(l+1)/(2r²) + (1/2) ψ_ps''(r)/ψ_ps(r)
```

### Unscreening

The ionic (bare) pseudopotential:
```
V_ps,ion(r) = V_ps,scr(r) - V_H[n_ps](r) - V_xc[n_ps](r)
```
where n_ps is the pseudo-valence density.

### Kleinman-Bylander Form

Non-local part in separable form:
```
V_nl = Σ_l |δV_l φ_l⟩ ⟨φ_l δV_l| / ⟨φ_l|δV_l|φ_l⟩
```
where δV_l = V_l - V_local

## Implementation Steps

### Step 1: Core/Valence Partition
```julia
struct AtomConfig
    Z::Float64
    core_orbitals::Vector{Tuple{Int,Int,Float64}}   # (n, l, occ) frozen
    valence_orbitals::Vector{Tuple{Int,Int,Float64}} # (n, l, occ) for PP
end
```

Example for Al:
- Core: 1s², 2s², 2p⁶ (10 electrons)
- Valence: 3s², 3p¹ (3 electrons)

### Step 2: Cutoff Radius Selection
```julia
function select_rc(grid, u_ae, l; method=:auto)
    # Options:
    # - Beyond outermost node
    # - At wavefunction maximum
    # - User-specified
    # Typical: r_c ≈ 1.0-2.5 a.u. for valence orbitals
end
```

### Step 3: Troullier-Martins Pseudo-wavefunction
```julia
function troullier_martins_pswf(grid, u_ae, E_ae, l, rc)
    # 1. Compute AE values at rc: ψ(rc), ψ'(rc), ψ''(rc), etc.
    # 2. Compute norm inside rc: ∫₀^rc |ψ_AE|² dr
    # 3. Set up system for TM coefficients c₀, c₂, ..., c₁₂
    # 4. Solve nonlinear system (norm conservation is nonlinear)
    # 5. Construct ψ_ps = r^(l+1) exp(p(r)) for r < rc
    # 6. Match to ψ_AE for r ≥ rc
    return u_ps
end
```

### Step 4: Screened Pseudopotential by Inversion
```julia
function invert_schrodinger(grid, u_ps, E, l)
    # V_scr(r) = E - l(l+1)/(2r²) + (1/2) u''(r)/u(r)
    # Handle numerical derivatives carefully near origin
    return V_scr
end
```

### Step 5: Unscreening
```julia
function unscreen_potential(grid, V_scr, n_ps)
    V_H = solve_poisson(grid, n_ps)
    V_xc = [lda_pz81(n)[1] for n in n_ps]
    V_ion = V_scr - V_H - V_xc
    return V_ion
end
```

### Step 6: Construct Full Pseudopotential
```julia
struct NormConservingPP
    Z_ion::Float64              # Ionic charge (Z - N_core)
    r_c::Vector{Float64}        # Cutoff radii for each l
    V_local::Vector{Float64}    # Local potential (usually highest l)
    V_l::Vector{Vector{Float64}} # Semi-local potentials for each l
    projectors::Vector{Vector{Float64}}  # KB projectors
    E_kb::Vector{Float64}       # KB energies ⟨φ|δV|φ⟩
end
```

### Step 7: Kleinman-Bylander Projectors
```julia
function construct_kb_projectors(grid, V_local, V_l, phi_l)
    # δV_l = V_l - V_local
    # |p_l⟩ = δV_l |φ_l⟩
    # E_kb = ⟨φ_l|δV_l|φ_l⟩
    # Check for ghost states: E_kb should not change sign
end
```

### Step 8: Validation
```julia
function validate_pseudopotential(pp, grid, ae_results)
    # 1. Solve pseudo-atom SCF
    # 2. Compare eigenvalues with AE
    # 3. Check log-derivative transferability at different energies
    # 4. Check for ghost states in KB form
end
```

## Data Structures

```julia
# Main generation function
function generate_ncpp(
    Z::Float64,
    core_config::Vector{Tuple{Int,Int,Float64}},
    valence_config::Vector{Tuple{Int,Int,Float64}};
    rc::Union{Nothing, Dict{Int,Float64}} = nothing,  # l -> r_c
    local_channel::Int = -1,  # -1 = auto (highest l)
    grid_params = (Rp=0.005, r0=1e-6, r_max=30.0, N=5000)
)
    # 1. Run AE calculation for full atom
    # 2. For each valence l-channel:
    #    - Select r_c
    #    - Generate TM pseudo-wavefunction
    #    - Invert to get screened potential
    # 3. Compute pseudo-density
    # 4. Unscreen all potentials
    # 5. Choose local potential
    # 6. Construct KB projectors
    # 7. Validate
    return NormConservingPP(...)
end
```

## File Output

Support standard formats:
- UPF (Quantum ESPRESSO)
- PSP8 (ABINIT)
- Simple text format for testing

## Testing Plan

1. **Hydrogen**: Simple case, no core
2. **Carbon**: 1s² core, 2s²2p² valence
3. **Silicon**: [Ne] core, 3s²3p² valence
4. **Aluminum**: [Ne] core, 3s²3p¹ valence (our test case)

## Implementation Order

1. `select_rc()` - cutoff radius selection
2. `troullier_martins_pswf()` - TM pseudo-wavefunction generation
3. `invert_schrodinger()` - potential inversion
4. `unscreen_potential()` - unscreening
5. `NormConservingPP` struct and `generate_ncpp()` main function
6. `construct_kb_projectors()` - Kleinman-Bylander form
7. `validate_pseudopotential()` - validation tests
8. File I/O for standard formats

## References

1. Hamann, Schlüter, Chiang, PRL 43, 1494 (1979) - Original NC conditions
2. Troullier, Martins, PRB 43, 1993 (1991) - TM scheme
3. Kleinman, Bylander, PRL 48, 1425 (1982) - KB separable form
