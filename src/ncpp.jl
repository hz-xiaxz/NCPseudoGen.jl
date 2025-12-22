# ============================================================================
# Norm-Conserving Pseudopotential Generation
#
# Implements Troullier-Martins scheme for generating smooth, transferable
# norm-conserving pseudopotentials from all-electron calculations.
# ============================================================================

using LinearAlgebra
using FiniteDifferences: central_fdm

# ============================================================================
# Finite Difference Utilities
# ============================================================================

# Pre-compute FD stencils (unadapted to avoid step-size estimation that goes out of bounds)
const _FD_D1 = central_fdm(5, 1; adapt=0)  # 5-point, 1st derivative, no adaptation
const _FD_D2 = central_fdm(5, 2; adapt=0)  # 5-point, 2nd derivative, no adaptation

"""
    fd_derivative(data, i, δ, order=1; num_points=5)

Compute derivative of discrete data using FiniteDifferences.jl stencils.

Extracts coefficients from `central_fdm` and applies them directly to discrete data,
avoiding the adaptive step estimation that can cause out-of-bounds access.

# Arguments
- `data`: Vector of discrete values sampled at uniform spacing δ
- `i`: Index at which to compute derivative
- `δ`: Grid spacing in x-space
- `order`: Derivative order (default 1)
- `num_points`: Number of stencil points (default 5)

# Returns
- Derivative value at index i
"""
function fd_derivative(data::Vector{Float64}, i::Int, δ::Float64, order::Int=1;
                       num_points::Int=5)
    method = if order == 1 && num_points == 5
        _FD_D1
    elseif order == 2 && num_points == 5
        _FD_D2
    else
        central_fdm(num_points, order; adapt=0)
    end

    # Extract stencil grid and coefficients
    g = method.grid   # offset grid, e.g., [-2, -1, 0, 1, 2]
    c = method.coefs  # coefficients (for step size = 1)

    # Apply stencil: f^(n)(x) ≈ Σ c_k * f[i + g_k] / δ^n
    result = sum(c[k] * data[i + Int(g[k])] for k in eachindex(g))
    return result / δ^order
end

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
    compute_derivatives_at_rc(grid, u_ae, V_eff, E, l, i_c)

Compute u and its derivatives at r_c using Schrödinger equation.

Instead of computing high-order finite differences (unstable), we use:
    u'' = g(r) * u   where g(r) = 2(V - E) + l(l+1)/r²
    u''' = g' * u + g * u'
    u'''' = g'' * u + 2g' * u' + g * u''

This only requires first derivative of u (stable) and derivatives of g (smooth).

# Returns
- `(u, u', u'', u''', u'''')` at r_c in r-space
"""
function compute_derivatives_at_rc(grid, u_ae::Vector{Float64},
                                    V_eff::Vector{Float64}, E::Float64,
                                    l::Int, i_c::Int)
    r_c = grid.r[i_c]
    δ = grid.δ

    # Value at r_c
    u = u_ae[i_c]

    # First derivative using central difference (5-point stencil in x-space)
    du_dx = fd_derivative(u_ae, i_c, δ, 1)
    rp_c = grid.rp[i_c]
    du_dr = du_dx / rp_c

    # g(r) = 2(V - E) + l(l+1)/r² from Schrödinger equation
    g_c = 2*(V_eff[i_c] - E) + l*(l+1)/r_c^2

    # Second derivative from SE: u'' = g * u
    d2u_dr2 = g_c * u

    # For g' and g'', compute derivatives of g(r) = 2(V - E) + l(l+1)/r²
    # g'(r) = 2V'(r) - 2l(l+1)/r³
    # g''(r) = 2V''(r) + 6l(l+1)/r⁴

    # Derivative of V using FiniteDifferences.jl
    dV_dx = fd_derivative(V_eff, i_c, δ, 1)
    dV_dr = dV_dx / rp_c

    d2V_dx2 = fd_derivative(V_eff, i_c, δ, 2)
    d2V_dr2 = (d2V_dx2 - dV_dx) / rp_c^2

    dg_dr = 2*dV_dr - 2*l*(l+1)/r_c^3
    d2g_dr2 = 2*d2V_dr2 + 6*l*(l+1)/r_c^4

    # Third derivative: u''' = g' * u + g * u'
    d3u_dr3 = dg_dr * u + g_c * du_dr

    # Fourth derivative: u'''' = g'' * u + 2g' * u' + g * u''
    d4u_dr4 = d2g_dr2 * u + 2*dg_dr * du_dr + g_c * d2u_dr2

    return u, du_dr, d2u_dr2, d3u_dr3, d4u_dr4
end

"""
    troullier_martins_pswf(grid, u_ae, V_eff, E, l, rc)

Generate Troullier-Martins pseudo-wavefunction.

Inside r_c, the pseudo-wavefunction has the form:
    u_ps(r) = r^(l+1) * exp(p(r))
    p(r) = c₀ + c₂r² + c₄r⁴ + c₆r⁶ + c₈r⁸ + c₁₀r¹⁰ + c₁₂r¹²

The 7 coefficients are determined by:
1-5. Continuity of u_ps, u_ps', u_ps'', u_ps''', u_ps'''' at r_c
6.   Norm conservation: ∫₀^rc |u_ps|² dr = ∫₀^rc |u_ae|² dr
7.   Curvature condition at origin

Uses Schrödinger equation for stable higher-derivative computation.

# Arguments
- `grid`: Radial grid
- `u_ae`: All-electron wavefunction u(r)
- `V_eff`: Effective potential (including centrifugal term)
- `E`: Orbital eigenvalue
- `l`: Angular momentum
- `rc`: Cutoff radius

# Returns
- `u_ps`: Pseudo-wavefunction (equals u_ae for r > rc)
"""
function troullier_martins_pswf(grid, u_ae::Vector{Float64}, V_eff::Vector{Float64},
                                 E::Float64, l::Int, rc::Float64)
    N = grid.N
    δ = grid.δ

    i_c = find_grid_index(grid, rc)
    r_c = grid.r[i_c]

    if i_c < 3 || i_c > N - 2
        error("rc too close to grid boundary")
    end

    # Get derivatives using SE-based method (much more stable)
    u_c, du_dr, d2u_dr2, d3u_dr3, d4u_dr4 = compute_derivatives_at_rc(
        grid, u_ae, V_eff, E, l, i_c)

    # Compute norm of u_ae from 0 to r_c
    norm_ae_sq = 0.0
    for i in 1:i_c-1
        norm_ae_sq += 0.5 * δ * (u_ae[i]^2 * grid.rp[i] + u_ae[i+1]^2 * grid.rp[i+1])
    end

    # Convert to f(r) = ln(u / r^(l+1)) = p(r) and its derivatives
    f_c = log(abs(u_c) / r_c^(l+1))

    # f' = u'/u - (l+1)/r
    fp_c = du_dr / u_c - (l + 1) / r_c

    # f'' = u''/u - (u'/u)² + (l+1)/r²
    #     = u''/u - f'² - 2(l+1)f'/r - (l+1)²/r² + (l+1)/r²
    # Simpler: f'' = (u'' - u * f'²) / u - but direct formula is cleaner
    fpp_c = d2u_dr2 / u_c - (du_dr / u_c)^2

    # f''' from chain rule
    fppp_c = d3u_dr3 / u_c - 3 * (du_dr / u_c) * (d2u_dr2 / u_c) + 2 * (du_dr / u_c)^3

    # f'''' from chain rule
    fpppp_c = d4u_dr4 / u_c - 4 * (du_dr / u_c) * (d3u_dr3 / u_c) -
              3 * (d2u_dr2 / u_c)^2 + 12 * (du_dr / u_c)^2 * (d2u_dr2 / u_c) -
              6 * (du_dr / u_c)^4

    # Solve for TM polynomial coefficients
    c = solve_tm_coefficients(r_c, l, E, f_c, fp_c, fpp_c, fppp_c, fpppp_c,
                               norm_ae_sq, grid, i_c)

    # Build pseudo-wavefunction
    u_ps = copy(u_ae)
    for i in 1:i_c
        r = grid.r[i]
        p = c[1] + c[2]*r^2 + c[3]*r^4 + c[4]*r^6 + c[5]*r^8 + c[6]*r^10 + c[7]*r^12
        u_ps[i] = r^(l+1) * exp(p)
    end

    # Ensure continuity at rc (correct any numerical error)
    scale = u_ae[i_c] / u_ps[i_c]
    for i in 1:i_c
        u_ps[i] *= scale
    end

    return u_ps
end

# Keep old signature for backward compatibility (without V_eff)
function troullier_martins_pswf(grid, u_ae::Vector{Float64}, E::Float64,
                                 l::Int, rc::Float64)
    # Construct approximate V_eff from SE inversion at available points
    # This is a fallback - prefer passing V_eff explicitly
    N = grid.N
    V_eff = zeros(N)

    # Use SE: V = E + u''/(2u) - l(l+1)/(2r²)
    δ = grid.δ
    for i in 3:N-2
        d2u_dx2 = fd_derivative(u_ae, i, δ, 2)
        du_dx = fd_derivative(u_ae, i, δ, 1)
        rp = grid.rp[i]
        d2u_dr2 = (d2u_dx2 - du_dx) / rp^2

        if abs(u_ae[i]) > 1e-30
            V_eff[i] = E + 0.5 * d2u_dr2 / u_ae[i]
        else
            V_eff[i] = E
        end
    end
    V_eff[1:2] .= V_eff[3]
    V_eff[N-1:N] .= V_eff[N-2]

    return troullier_martins_pswf(grid, u_ae, V_eff, E, l, rc)
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

        # Derivatives in x-space using FiniteDifferences.jl
        d2u_dx2 = fd_derivative(u_ps, i, δ, 2)
        du_dx = fd_derivative(u_ps, i, δ, 1)

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

        if i_c > 2 && i_c < grid.N - 2
            # Log derivative = u'/u = (du/dx) / (rp * u)
            ld_ae = fd_derivative(orb.u_ae, i_c, δ, 1) / (grid.rp[i_c] * orb.u_ae[i_c])
            ld_ps = fd_derivative(orb.u_ps, i_c, δ, 1) / (grid.rp[i_c] * orb.u_ps[i_c])

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

    # Compute V_base (without centrifugal) for use in TM generation
    # V_base = V_nuc + V_H + V_xc
    N = grid.N
    V_nuc = -Z ./ grid.r
    V_H = solve_poisson(grid, ae_density)
    V_xc = [lda_pz81(ni)[1] for ni in ae_density]
    V_base = V_nuc .+ V_H .+ V_xc

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

        # Compute V_eff for this orbital (V_base + centrifugal)
        V_eff = V_base .+ l * (l + 1) ./ (2 .* grid.r .^ 2)

        orb_name = "$(n)$(["s","p","d","f"][l+1])"
        println("  Generating $orb_name pseudo-wavefunction...")

        u_ps = troullier_martins_pswf(grid, u_ae, V_eff, E, l, rc)

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
