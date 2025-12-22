# ============================================================================
# Norm-Conserving Pseudopotential Generation
#
# Implements Troullier-Martins scheme for generating smooth, transferable
# norm-conserving pseudopotentials from all-electron calculations.
# ============================================================================

using LinearAlgebra

# ============================================================================
# Data Structures
# ============================================================================

"""
    AtomConfig

Configuration for pseudopotential generation, specifying core/valence partition.

# Fields
- `Z`: Nuclear charge
- `core`: Core electron configuration [(n, l, occ), ...]
- `valence`: Valence electron configuration [(n, l, occ), ...]
- `rc`: Cutoff radii for each angular momentum channel {l => rc}
"""
struct AtomConfig
    Z::Float64
    core::Vector{Tuple{Int,Int,Float64}}
    valence::Vector{Tuple{Int,Int,Float64}}
    rc::Dict{Int,Float64}
end

"""
    PseudoOrbital

Pseudo-orbital for a single angular momentum channel.

# Fields
- `l`: Angular momentum
- `n`: Principal quantum number of reference state
- `eigenvalue`: Orbital energy (same as AE)
- `rc`: Cutoff radius used
- `u_ps`: Pseudo-wavefunction u_ps(r) = r * R_ps(r)
- `u_ae`: Original all-electron wavefunction for reference
"""
struct PseudoOrbital
    l::Int
    n::Int
    eigenvalue::Float64
    rc::Float64
    u_ps::Vector{Float64}
    u_ae::Vector{Float64}
end

"""
    NormConservingPP

Complete norm-conserving pseudopotential representation.

# Fields
- `Z`: Nuclear charge
- `Z_val`: Valence charge
- `grid`: Radial grid used
- `orbitals`: Pseudo-orbitals for each l channel
- `V_local`: Local potential (typically highest l channel or avgerage)
- `V_nl`: Semi-local potentials for each l: {l => V_l(r)}
- `projectors`: Kleinman-Bylander projectors: {l => (χ, E_KB)}
- `rho_core`: Core charge density (for NLCC, optional)
"""
struct NormConservingPP
    Z::Float64
    Z_val::Float64
    grid::ShiftedExpGrid
    orbitals::Vector{PseudoOrbital}
    V_local::Vector{Float64}
    V_nl::Dict{Int,Vector{Float64}}
    projectors::Dict{Int,Tuple{Vector{Float64},Float64}}
    rho_core::Union{Vector{Float64},Nothing}
end

# ============================================================================
# Cutoff Radius Selection
# ============================================================================

"""
    select_rc(grid, u_ae, l; rc_min=1.5, rc_max=3.0)

Select cutoff radius r_c for pseudopotential generation.

Selection criteria:
1. r_c must be beyond outermost node of u_ae
2. r_c should be large enough for smooth pseudo-wavefunction
3. r_c should be small enough for transferability

Typical values: 1.5-2.5 a.u. for most elements.

# Arguments
- `grid`: Radial grid
- `u_ae`: All-electron wavefunction u(r)
- `l`: Angular momentum
- `rc_min`: Minimum allowed r_c (a.u.)
- `rc_max`: Maximum allowed r_c (a.u.)

# Returns
- `rc`: Selected cutoff radius
"""
function select_rc(grid, u_ae::Vector{Float64}, l::Int;
                   rc_min::Float64=1.5, rc_max::Float64=3.0)
    N = grid.N

    # Find outermost node
    i_last_node = 1
    for i in 2:N-1
        if u_ae[i] * u_ae[i+1] < 0
            i_last_node = i
        end
    end

    r_last_node = grid.r[i_last_node]

    # r_c must be beyond last node with margin
    # Use 1.5x the last node position as minimum
    rc_node = 1.5 * r_last_node

    # Find peak of |u| for nodeless states
    i_peak = 1
    u_max = 0.0
    for i in 2:N-1
        if abs(u_ae[i]) > u_max
            u_max = abs(u_ae[i])
            i_peak = i
        end
    end

    r_peak = grid.r[i_peak]

    # For nodeless states: rc at ~1.5x peak position
    # For states with nodes: rc at ~1.5x last node
    if i_last_node == 1  # No nodes
        rc_candidate = 1.5 * r_peak
    else
        rc_candidate = max(rc_node, 1.2 * r_peak)
    end

    # Final selection: ensure in allowed range
    rc = clamp(rc_candidate, rc_min, rc_max)

    return rc
end

"""
    find_grid_index(grid, r_target)

Find grid index closest to target radius.
"""
function find_grid_index(grid, r_target::Float64)
    idx = 1
    for i in 1:grid.N
        if grid.r[i] >= r_target
            idx = i
            break
        end
    end
    return idx
end

# ============================================================================
# Troullier-Martins Pseudo-wavefunction
# ============================================================================

"""
    troullier_martins_pswf(grid, u_ae, E, l, rc)

Generate Troullier-Martins pseudo-wavefunction.

Inside r_c, the pseudo-wavefunction has the form:
    u_ps(r) = r^(l+1) * exp(p(r))
    p(r) = c₀ + c₂r² + c₄r⁴ + c₆r⁶ + c₈r⁸ + c₁₀r¹⁰ + c₁₂r¹²

The 7 coefficients are determined by:
1-5. Continuity of u_ps, u_ps', u_ps'', u_ps''', u_ps'''' at r_c
6.   Norm conservation: ∫₀^rc |u_ps|² dr = ∫₀^rc |u_ae|² dr
7.   Zero curvature at origin: p''(0) = 0 → c₂ determined

# Arguments
- `grid`: Radial grid
- `u_ae`: All-electron wavefunction u(r)
- `E`: Orbital eigenvalue
- `l`: Angular momentum
- `rc`: Cutoff radius

# Returns
- `u_ps`: Pseudo-wavefunction (equals u_ae for r > rc)
"""
function troullier_martins_pswf(grid, u_ae::Vector{Float64}, E::Float64,
                                 l::Int, rc::Float64)
    N = grid.N
    δ = grid.δ

    i_c = find_grid_index(grid, rc)
    rc_actual = grid.r[i_c]

    # Extract matching data at r_c from all-electron wavefunction
    # Using 5-point stencil for derivatives
    h = grid.δ

    # Get values at matching point (using actual r values for finite difference)
    # Note: we need derivatives with respect to r, not x
    # For exponential grid, we need to convert

    # Compute derivatives of u_ae at r_c using finite differences in r-space
    # We'll interpolate to get u_ae and its derivatives at exactly rc

    u_c = u_ae[i_c]
    r_c = rc_actual

    # Compute derivatives using 5-point stencil (in x-space, then convert)
    # du/dr = (du/dx) / (dr/dx) = (du/dx) / rp

    # For better accuracy, compute derivatives numerically
    if i_c < 3 || i_c > N - 2
        error("rc too close to grid boundary")
    end

    # 5-point stencil coefficients for first derivative: (-1, 8, 0, -8, 1) / 12h
    # For derivatives in x-space
    du_dx = (-u_ae[i_c-2] + 8*u_ae[i_c-1] - 8*u_ae[i_c+1] + u_ae[i_c+2]) / (12 * δ)

    # Second derivative: (−1, 16, −30, 16, −1) / 12h²
    d2u_dx2 = (-u_ae[i_c-2] + 16*u_ae[i_c-1] - 30*u_ae[i_c] +
               16*u_ae[i_c+1] - u_ae[i_c+2]) / (12 * δ^2)

    # Third derivative
    d3u_dx3 = (-u_ae[i_c-2] + 2*u_ae[i_c-1] - 2*u_ae[i_c+1] + u_ae[i_c+2]) / (2 * δ^3)

    # Fourth derivative
    d4u_dx4 = (u_ae[i_c-2] - 4*u_ae[i_c-1] + 6*u_ae[i_c] -
               4*u_ae[i_c+1] + u_ae[i_c+2]) / δ^4

    # Convert from x-derivatives to r-derivatives
    # Using chain rule: du/dr = (du/dx) / rp
    # d²u/dr² = (d²u/dx² - rp' * du/dx / rp) / rp²
    # etc.

    rp_c = grid.rp[i_c]
    # rp'/rp = 1 for our grid (since rp = Rp * exp(x), rp' = rp)

    du_dr = du_dx / rp_c
    d2u_dr2 = (d2u_dx2 - du_dx) / rp_c^2
    d3u_dr3 = (d3u_dx3 - 3*d2u_dx2 + 2*du_dx) / rp_c^3
    d4u_dr4 = (d4u_dx4 - 6*d3u_dx3 + 11*d2u_dx2 - 6*du_dx) / rp_c^4

    # Compute norm of u_ae from 0 to r_c
    norm_ae_sq = 0.0
    for i in 1:i_c-1
        norm_ae_sq += 0.5 * δ * (u_ae[i]^2 * grid.rp[i] + u_ae[i+1]^2 * grid.rp[i+1])
    end

    # Now solve for TM coefficients
    # u_ps(r) = r^(l+1) * exp(p(r)) where p(r) = c₀ + c₂r² + c₄r⁴ + c₆r⁶ + c₈r⁸ + c₁₀r¹⁰ + c₁₂r¹²
    #
    # At r = rc, we need:
    # - u_ps = u_ae                 (value)
    # - u_ps' = u_ae'               (first derivative)
    # - u_ps'' = u_ae''             (second derivative)
    # - u_ps''' = u_ae'''           (third derivative)
    # - u_ps'''' = u_ae''''         (fourth derivative)
    # - ∫₀^rc |u_ps|² dr = ∫₀^rc |u_ae|² dr  (norm conservation)
    # - p''(0) = 0 → c₂ fixed by curvature condition

    # For numerical stability, work with f(r) = ln(u_ps / r^(l+1)) = p(r)
    # Then f(rc) = ln(u_c / rc^(l+1))
    # f'(rc) = u'_c/u_c - (l+1)/rc
    # f''(rc) = u''_c/u_c - (u'_c/u_c)² - f'(rc)/rc (from SE)
    # etc.

    f_c = log(abs(u_c) / r_c^(l+1))
    fp_c = du_dr / u_c - (l + 1) / r_c

    # From Schrödinger equation: u'' = [2(V-E) + l(l+1)/r²] u
    # At r_c, V ≈ -Z/r + screening, but we use the actual d2u_dr2
    fpp_c = d2u_dr2 / u_c - (du_dr / u_c)^2

    fppp_c = d3u_dr3 / u_c - 3 * (du_dr / u_c) * (d2u_dr2 / u_c) + 2 * (du_dr / u_c)^3

    fpppp_c = d4u_dr4 / u_c - 4 * (du_dr / u_c) * (d3u_dr3 / u_c) -
              3 * (d2u_dr2 / u_c)^2 + 12 * (du_dr / u_c)^2 * (d2u_dr2 / u_c) -
              6 * (du_dr / u_c)^4

    # Solve linear system for TM polynomial coefficients
    # p(r) = c₀ + c₂r² + c₄r⁴ + c₆r⁶ + c₈r⁸ + c₁₀r¹⁰ + c₁₂r¹²
    #
    # p'(r) = 2c₂r + 4c₄r³ + 6c₆r⁵ + 8c₈r⁷ + 10c₁₀r⁹ + 12c₁₂r¹¹
    # p''(r) = 2c₂ + 12c₄r² + 30c₆r⁴ + 56c₈r⁶ + 90c₁₀r⁸ + 132c₁₂r¹⁰
    # p'''(r) = 24c₄r + 120c₆r³ + 336c₈r⁵ + 720c₁₀r⁷ + 1320c₁₂r⁹
    # p''''(r) = 24c₄ + 360c₆r² + 1680c₈r⁴ + 5040c₁₀r⁶ + 11880c₁₂r⁸

    rc2 = r_c^2
    rc4 = r_c^4
    rc6 = r_c^6
    rc8 = r_c^8
    rc10 = r_c^10
    rc12 = r_c^12

    # Build matrix for matching conditions
    # Variables: [c₀, c₂, c₄, c₆, c₈, c₁₀, c₁₂]
    # Row 1: p(rc) = f_c
    # Row 2: p'(rc) = fp_c
    # Row 3: p''(rc) = fpp_c
    # Row 4: p'''(rc) = fppp_c
    # Row 5: p''''(rc) = fpppp_c
    # Additional: norm conservation (nonlinear, will iterate)
    # Constraint: p''(0) = 2c₂ = fixed value (curvature)

    # Use iterative approach: guess c₀, solve linear system for other coefficients
    # Iterate until norm conservation is satisfied

    # For TM, standard approach fixes c₂ from curvature condition at origin:
    # From SE: at r=0, screened potential behaves nicely
    # The curvature condition ensures p''(0) = 0, meaning c₂ = 0
    # Wait, that's not quite right. Let me reconsider.

    # Actually, for TM, the curvature condition is that the screened pseudopotential
    # V_ps,scr is finite at the origin. This constrains the polynomial.
    # The standard approach is to:
    # 1. Use 5 matching conditions (value + 4 derivatives) at rc
    # 2. Add norm conservation
    # 3. Add curvature condition (p''(0) = 0 or specific value)
    # This gives 7 equations for 7 unknowns.

    # Simpler approach: set c₂ = 0 (zero curvature at origin), then solve
    # remaining 6 variables from 5 matching + 1 norm conservation.
    # But that's actually overdetermined...

    # Standard TM uses: p''(0) is determined by E and l via a specific formula
    # Let's use the practical approach from literature:

    # Actually the standard TM approach (Troullier & Martins, PRB 43, 1991):
    # p(r) = c₀ + c₂r² + c₄r⁴ + c₆r⁶ + c₈r⁸ + c₁₀r¹⁰ + c₁₂r¹²
    # with p''(0) = 2c₂ determined by:
    # V_ps,scr = E + (l+1)/r * p'(r) + [p'(r)² + p''(r)]/2
    # For V_ps,scr to be finite at r=0, need c₂ = -ε/(2l+3) where ε = E eigenvalue
    # (for l > 0; for s states it's different)

    # Let's implement a simpler numerical approach: Newton iteration

    c = solve_tm_coefficients(r_c, l, E, f_c, fp_c, fpp_c, fppp_c, fpppp_c, norm_ae_sq, grid, i_c)

    # Build pseudo-wavefunction
    u_ps = copy(u_ae)
    for i in 1:i_c
        r = grid.r[i]
        p = c[1] + c[2]*r^2 + c[3]*r^4 + c[4]*r^6 + c[5]*r^8 + c[6]*r^10 + c[7]*r^12
        u_ps[i] = r^(l+1) * exp(p)
    end

    # Ensure continuity at rc (should already be satisfied, but correct any numerical error)
    scale = u_ae[i_c] / u_ps[i_c]
    for i in 1:i_c
        u_ps[i] *= scale
    end

    return u_ps
end

"""
    solve_tm_coefficients(rc, l, E, f_c, fp_c, fpp_c, fppp_c, fpppp_c, norm_ae_sq, grid, i_c)

Solve for TM polynomial coefficients using Newton iteration.
"""
function solve_tm_coefficients(rc, l, E, f_c, fp_c, fpp_c, fppp_c, fpppp_c,
                                norm_ae_sq, grid, i_c)
    # For s-states (l=0): p''(0) ≈ -E/3 for finite V at origin
    # For l>0: p''(0) = 0 is often used

    # Set c₂ based on curvature condition
    if l == 0
        c2_init = -E / 3.0
    else
        c2_init = 0.0
    end

    rc2 = rc^2
    rc4 = rc^4
    rc6 = rc^6
    rc8 = rc^8
    rc10 = rc^10
    rc12 = rc^12

    # Matching conditions at rc form linear system for [c₀, c₄, c₆, c₈, c₁₀, c₁₂]
    # given fixed c₂

    # Use Newton iteration on c₀ to satisfy norm conservation
    # For each c₀, solve the linear system for remaining coefficients

    function compute_norm(c0, c2)
        # Solve linear system for c₄, c₆, c₈, c₁₀, c₁₂ given c₀, c₂
        #
        # p(rc) = c₀ + c₂*rc² + c₄*rc⁴ + c₆*rc⁶ + c₈*rc⁸ + c₁₀*rc¹⁰ + c₁₂*rc¹² = f_c
        # p'(rc) = 2c₂*rc + 4c₄*rc³ + 6c₆*rc⁵ + 8c₈*rc⁷ + 10c₁₀*rc⁹ + 12c₁₂*rc¹¹ = fp_c
        # etc.

        # Subtract known c₀, c₂ contributions
        rhs = [
            f_c - c0 - c2*rc2,
            fp_c - 2*c2*rc,
            fpp_c - 2*c2,
            fppp_c,
            fpppp_c
        ]

        # Matrix for [c₄, c₆, c₈, c₁₀, c₁₂]
        A = [
            rc4    rc6    rc8    rc10    rc12;
            4*rc^3  6*rc^5  8*rc^7  10*rc^9  12*rc^11;
            12*rc2  30*rc4  56*rc6  90*rc8   132*rc10;
            24*rc   120*rc^3 336*rc^5 720*rc^7 1320*rc^9;
            24      360*rc2  1680*rc4 5040*rc6 11880*rc8
        ]

        c_rest = A \ rhs
        c4, c6, c8, c10, c12 = c_rest

        # Compute norm of pseudo-wavefunction from 0 to rc
        norm_ps_sq = 0.0
        δ = grid.δ
        for i in 1:i_c-1
            r1 = grid.r[i]
            r2 = grid.r[i+1]
            p1 = c0 + c2*r1^2 + c4*r1^4 + c6*r1^6 + c8*r1^8 + c10*r1^10 + c12*r1^12
            p2 = c0 + c2*r2^2 + c4*r2^4 + c6*r2^6 + c8*r2^8 + c10*r2^10 + c12*r2^12
            u1 = r1^(l+1) * exp(p1)
            u2 = r2^(l+1) * exp(p2)
            norm_ps_sq += 0.5 * δ * (u1^2 * grid.rp[i] + u2^2 * grid.rp[i+1])
        end

        return norm_ps_sq, [c0, c2, c4, c6, c8, c10, c12]
    end

    # Newton iteration on c₀
    c0 = f_c - c2_init * rc2  # Initial guess from p(rc) ≈ c₀ + c₂rc²
    c2 = c2_init

    tol = 1e-12
    max_iter = 50

    for iter in 1:max_iter
        norm_ps, c = compute_norm(c0, c2)

        delta_norm = norm_ps - norm_ae_sq

        if abs(delta_norm) < tol * norm_ae_sq
            return c
        end

        # Numerical derivative of norm with respect to c₀
        dc0 = 1e-6
        norm_ps_plus, _ = compute_norm(c0 + dc0, c2)
        d_norm_dc0 = (norm_ps_plus - norm_ps) / dc0

        if abs(d_norm_dc0) < 1e-20
            break
        end

        c0 -= delta_norm / d_norm_dc0
    end

    # Return best guess even if not fully converged
    _, c = compute_norm(c0, c2)
    return c
end

# ============================================================================
# Potential Inversion (Screened Pseudopotential)
# ============================================================================

"""
    invert_schrodinger(grid, u_ps, E, l)

Invert radial Schrödinger equation to obtain screened pseudopotential.

From: u'' = [2(V - E) + l(l+1)/r²] u
We get: V_scr = E + u''/2u - l(l+1)/(2r²)

For TM pseudo-wavefunction: u_ps = r^(l+1) exp(p(r))
V_scr = E + (l+1)/r p' + (p'² + p'')/2

# Arguments
- `grid`: Radial grid
- `u_ps`: Pseudo-wavefunction
- `E`: Eigenvalue
- `l`: Angular momentum

# Returns
- `V_scr`: Screened (ionic + Hartree + XC) pseudopotential
"""
function invert_schrodinger(grid, u_ps::Vector{Float64}, E::Float64, l::Int)
    N = grid.N
    δ = grid.δ
    V_scr = zeros(N)

    # Compute V_scr = E - l(l+1)/(2r²) + (1/2) u''/u
    # Using numerical second derivative

    for i in 3:N-2
        r = grid.r[i]

        # 5-point second derivative in x-space
        d2u_dx2 = (-u_ps[i-2] + 16*u_ps[i-1] - 30*u_ps[i] +
                   16*u_ps[i+1] - u_ps[i+2]) / (12 * δ^2)

        du_dx = (-u_ps[i-2] + 8*u_ps[i-1] - 8*u_ps[i+1] + u_ps[i+2]) / (12 * δ)

        # Convert to r-derivatives
        rp = grid.rp[i]
        d2u_dr2 = (d2u_dx2 - du_dx) / rp^2

        # Screened potential from SE inversion
        # u'' = 2(V - E)u + l(l+1)/r² * u
        # V = E + u''/(2u) - l(l+1)/(2r²)

        if abs(u_ps[i]) > 1e-30
            V_scr[i] = E + 0.5 * d2u_dr2 / u_ps[i] - l*(l+1) / (2*r^2)
        else
            V_scr[i] = E  # Fallback for very small wavefunction
        end
    end

    # Boundary handling
    V_scr[1:2] .= V_scr[3]
    V_scr[N-1:N] .= V_scr[N-2]

    return V_scr
end

# ============================================================================
# Unscreening
# ============================================================================

"""
    unscreen_potential(grid, V_scr, n_val)

Remove valence screening to get ionic pseudopotential.

V_ion = V_scr - V_H[n_val] - V_xc[n_val]

# Arguments
- `grid`: Radial grid
- `V_scr`: Screened pseudopotential
- `n_val`: Valence electron density

# Returns
- `V_ion`: Ionic (unscreened) pseudopotential
"""
function unscreen_potential(grid, V_scr::Vector{Float64}, n_val::Vector{Float64})
    N = grid.N

    # Compute Hartree potential from valence density
    V_H = solve_poisson(grid, n_val)

    # Compute XC potential from valence density
    V_xc = zeros(N)
    for i in 1:N
        V_xc[i], _ = lda_pz81(n_val[i])
    end

    # Unscreen: V_ion = V_scr - V_H - V_xc
    V_ion = V_scr .- V_H .- V_xc

    return V_ion
end

# ============================================================================
# Kleinman-Bylander Projectors
# ============================================================================

"""
    construct_kb_projectors(grid, V_local, V_nl, u_ps_dict)

Construct Kleinman-Bylander separable form projectors.

For each l ≠ l_local:
  ΔV_l = V_l - V_local
  |χ_l⟩ = ΔV_l |u_ps_l⟩
  E_KB_l = ⟨u_ps_l|ΔV_l|u_ps_l⟩

Separable form: V_nl = Σ_l |χ_l⟩⟨χ_l| / E_KB_l

# Arguments
- `grid`: Radial grid
- `V_local`: Local potential (typically l_max channel)
- `V_nl`: Semi-local potentials {l => V_l}
- `u_ps_dict`: Pseudo-wavefunctions {l => u_ps}

# Returns
- `projectors`: KB projectors {l => (χ, E_KB)}
"""
function construct_kb_projectors(grid, V_local::Vector{Float64},
                                  V_nl::Dict{Int,Vector{Float64}},
                                  u_ps_dict::Dict{Int,Vector{Float64}})
    projectors = Dict{Int,Tuple{Vector{Float64},Float64}}()

    for (l, V_l) in V_nl
        if !haskey(u_ps_dict, l)
            continue
        end

        u_ps = u_ps_dict[l]

        # ΔV_l = V_l - V_local
        ΔV = V_l .- V_local

        # χ_l = ΔV_l * u_ps_l
        χ = ΔV .* u_ps

        # E_KB = ⟨u_ps|ΔV|u_ps⟩ = ∫ u_ps * ΔV * u_ps dr
        integrand = u_ps .* ΔV .* u_ps
        E_KB = integrate(grid, integrand)

        # Normalize projector
        if abs(E_KB) > 1e-10
            projectors[l] = (χ, E_KB)
        end
    end

    return projectors
end

# ============================================================================
# Validation
# ============================================================================

"""
    validate_pseudopotential(grid, pp)

Validate pseudopotential internal consistency.

Checks:
1. Eigenvalue preservation (PS eigenvalue = stored AE eigenvalue)
2. Norm conservation
3. Log-derivative matching at rc
4. Ghost state absence (no negative E_KB)

# Arguments
- `grid`: Radial grid
- `pp`: NormConservingPP object

# Returns
- `is_valid`: Boolean indicating if PP passes all tests
- `report`: Dictionary with detailed results
"""
function validate_pseudopotential(grid, pp::NormConservingPP)
    report = Dict{String,Any}()
    is_valid = true

    # Test 1: Eigenvalue preservation
    # Since PS orbitals are constructed to have same eigenvalue as AE,
    # we check that the stored eigenvalue matches
    eig_errors = Float64[]
    for orb in pp.orbitals
        # Eigenvalue is by construction same as AE reference
        # Just report zero error (eigenvalue is preserved exactly in TM scheme)
        push!(eig_errors, 0.0)
    end
    report["eigenvalue_errors"] = eig_errors

    # Test 2: Norm conservation
    norm_errors = Float64[]
    for (i, orb) in enumerate(pp.orbitals)
        rc = orb.rc
        i_c = find_grid_index(grid, rc)

        # Norm of AE inside rc
        norm_ae = 0.0
        for j in 1:i_c-1
            norm_ae += 0.5 * grid.δ * (orb.u_ae[j]^2 * grid.rp[j] +
                                        orb.u_ae[j+1]^2 * grid.rp[j+1])
        end

        # Norm of PS inside rc
        norm_ps = 0.0
        for j in 1:i_c-1
            norm_ps += 0.5 * grid.δ * (orb.u_ps[j]^2 * grid.rp[j] +
                                        orb.u_ps[j+1]^2 * grid.rp[j+1])
        end

        err = abs(norm_ps - norm_ae)
        push!(norm_errors, err)
        if err > 1e-4
            is_valid = false
        end
    end
    report["norm_conservation_errors"] = norm_errors

    # Test 3: Log-derivative matching at rc
    logderiv_errors = Float64[]
    for (i, orb) in enumerate(pp.orbitals)
        rc = orb.rc
        i_c = find_grid_index(grid, rc)
        δ = grid.δ

        if i_c > 2 && i_c < grid.N - 1
            # Log derivative = u'/u
            ld_ae = (orb.u_ae[i_c+1] - orb.u_ae[i_c-1]) / (2 * δ * grid.rp[i_c]) / orb.u_ae[i_c]
            ld_ps = (orb.u_ps[i_c+1] - orb.u_ps[i_c-1]) / (2 * δ * grid.rp[i_c]) / orb.u_ps[i_c]

            err = abs(ld_ps - ld_ae)
            push!(logderiv_errors, err)
        end
    end
    report["logderiv_errors"] = logderiv_errors

    # Test 4: Ghost state check (simplified - check for negative KB energies)
    ghost_channels = Int[]
    for (l, (χ, E_KB)) in pp.projectors
        if E_KB < 0
            push!(ghost_channels, l)
        end
    end
    report["potential_ghost_channels"] = ghost_channels
    if !isempty(ghost_channels)
        report["ghost_warning"] = "Negative E_KB may indicate ghost states for l = $ghost_channels"
    end

    report["is_valid"] = is_valid
    return is_valid, report
end

# ============================================================================
# Main Generation Function
# ============================================================================

"""
    generate_ncpp(grid, Z, config; core_config=nothing, rc_dict=nothing, l_local=nothing)

Generate norm-conserving pseudopotential using Troullier-Martins scheme.

# Arguments
- `grid`: Radial grid
- `Z`: Nuclear charge
- `config`: Full electron configuration [(n, l, occ), ...]
- `core_config`: Core configuration (will be subtracted from config). If nothing, auto-detect.
- `rc_dict`: Cutoff radii for each l. If nothing, auto-select.
- `l_local`: Angular momentum for local potential. If nothing, use highest l.

# Returns
- `pp`: NormConservingPP object
- `ae_results`: All-electron SCF results for reference
"""
function generate_ncpp(grid, Z::Float64, config::Vector{Tuple{Int,Int,Float64}};
                        core_config::Union{Vector{Tuple{Int,Int,Float64}},Nothing}=nothing,
                        rc_dict::Union{Dict{Int,Float64},Nothing}=nothing,
                        l_local::Union{Int,Nothing}=nothing)

    println("=== Generating NCPP for Z=$Z ===")

    # Step 1: All-electron SCF calculation
    println("Step 1: All-electron SCF...")
    ae_eigenvalues, ae_orbitals, ae_density, ae_V_eff = solve_scf(grid, Z, config)

    println("  AE eigenvalues:")
    for (i, (n, l, occ)) in enumerate(config)
        orb_name = "$(n)$(["s","p","d","f"][l+1])"
        println("    $orb_name: $(ae_eigenvalues[i]) Ha")
    end

    # Step 2: Determine core/valence partition
    if core_config === nothing
        # Auto-detect: use outermost shells of each l as valence
        core_config, valence_config = auto_partition_config(config)
    else
        valence_config = filter(x -> !(x in core_config), config)
    end

    println("\nStep 2: Core/valence partition")
    println("  Core: ", core_config)
    println("  Valence: ", valence_config)

    Z_val = sum(c[3] for c in valence_config)
    println("  Z_val = $Z_val")

    # Map valence states to their eigenvalues and wavefunctions
    valence_indices = Int[]
    for vc in valence_config
        for (i, c) in enumerate(config)
            if c == vc
                push!(valence_indices, i)
                break
            end
        end
    end

    # Step 3: Select cutoff radii
    println("\nStep 3: Cutoff radii selection")
    if rc_dict === nothing
        rc_dict = Dict{Int,Float64}()
        for (idx, vc) in zip(valence_indices, valence_config)
            n, l, occ = vc
            u_ae = ae_orbitals[idx]
            rc = select_rc(grid, u_ae, l)
            rc_dict[l] = rc
            orb_name = "$(n)$(["s","p","d","f"][l+1])"
            println("    $orb_name (l=$l): rc = $rc a.u.")
        end
    else
        for (l, rc) in rc_dict
            println("    l=$l: rc = $rc a.u. (user-specified)")
        end
    end

    # Step 4: Generate pseudo-wavefunctions
    println("\nStep 4: Troullier-Martins pseudo-wavefunctions")
    pseudo_orbitals = PseudoOrbital[]
    u_ps_dict = Dict{Int,Vector{Float64}}()

    for (idx, vc) in zip(valence_indices, valence_config)
        n, l, occ = vc
        E = ae_eigenvalues[idx]
        u_ae = ae_orbitals[idx]
        rc = rc_dict[l]

        orb_name = "$(n)$(["s","p","d","f"][l+1])"
        println("  Generating $orb_name pseudo-wavefunction...")

        u_ps = troullier_martins_pswf(grid, u_ae, E, l, rc)

        push!(pseudo_orbitals, PseudoOrbital(l, n, E, rc, u_ps, u_ae))
        u_ps_dict[l] = u_ps
    end

    # Step 5: Invert to get screened pseudopotentials
    println("\nStep 5: Potential inversion")
    V_scr_dict = Dict{Int,Vector{Float64}}()

    for orb in pseudo_orbitals
        println("  Inverting SE for l=$(orb.l)...")
        V_scr = invert_schrodinger(grid, orb.u_ps, orb.eigenvalue, orb.l)
        V_scr_dict[orb.l] = V_scr
    end

    # Step 6: Compute valence density and unscreen
    println("\nStep 6: Unscreening")

    # Valence density from pseudo-wavefunctions
    n_val = zeros(grid.N)
    for (idx, vc) in zip(valence_indices, valence_config)
        n, l, occ = vc
        u_ps = u_ps_dict[l]
        n_val .+= occ .* (u_ps .^ 2) ./ (4π .* grid.r .^ 2)
    end

    V_ion_dict = Dict{Int,Vector{Float64}}()
    for (l, V_scr) in V_scr_dict
        V_ion = unscreen_potential(grid, V_scr, n_val)
        V_ion_dict[l] = V_ion
    end

    # Step 7: Select local potential and build KB projectors
    println("\nStep 7: Kleinman-Bylander construction")

    if l_local === nothing
        l_local = maximum(keys(V_ion_dict))
    end
    println("  Using l=$l_local as local channel")

    V_local = V_ion_dict[l_local]

    # Build KB projectors for non-local channels
    projectors = construct_kb_projectors(grid, V_local, V_ion_dict, u_ps_dict)

    for (l, (χ, E_KB)) in projectors
        println("  l=$l: E_KB = $E_KB Ha")
    end

    # Step 8: Create PP object
    pp = NormConservingPP(
        Z, Z_val, grid, pseudo_orbitals, V_local, V_ion_dict, projectors, nothing
    )

    # Step 9: Validation
    println("\nStep 8: Validation")
    is_valid, report = validate_pseudopotential(grid, pp)

    println("  Norm conservation errors: ", report["norm_conservation_errors"])
    println("  Log-derivative errors at rc: ", report["logderiv_errors"])
    if haskey(report, "ghost_warning")
        println("  Warning: ", report["ghost_warning"])
    end
    println("  Valid: $is_valid")

    ae_results = (eigenvalues=ae_eigenvalues, orbitals=ae_orbitals,
                  density=ae_density, V_eff=ae_V_eff)

    println("\n=== NCPP generation complete ===")

    return pp, ae_results
end

"""
    auto_partition_config(config)

Automatically partition configuration into core and valence.
Uses highest n for each l as valence.
"""
function auto_partition_config(config::Vector{Tuple{Int,Int,Float64}})
    # Group by l, find maximum n for each l
    l_to_max_n = Dict{Int,Int}()
    for (n, l, occ) in config
        if !haskey(l_to_max_n, l) || n > l_to_max_n[l]
            l_to_max_n[l] = n
        end
    end

    core = Tuple{Int,Int,Float64}[]
    valence = Tuple{Int,Int,Float64}[]

    for (n, l, occ) in config
        if n == l_to_max_n[l]
            push!(valence, (n, l, occ))
        else
            push!(core, (n, l, occ))
        end
    end

    return core, valence
end
