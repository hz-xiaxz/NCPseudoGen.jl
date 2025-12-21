"""
    Inward-Outward Numerov Solver for Radial Schrödinger Equation

The Numerov method solves equations of the form: y''(x) = f(x) * y(x)
with 4th-order accuracy.

For the radial Schrödinger equation (in atomic units):
    u''(r) = [2(V(r) - E) + l(l+1)/r²] * u(r) = g(r) * u(r)
    
On a ShiftedExpGrid: r = Rp*(exp(x)-1) + r0, dr/dx = rp = Rp*exp(x)

Using substitution y = u/sqrt(rp) transforms to:
    y''(x) = F(x) * y(x)
where F = rp² * g(r) + 1/4  (since rp'/rp = 1 and rp''/rp = 1)
"""

using Roots

# ============================================================================
# Integration helper
# ============================================================================
"""
    integrate(grid, f)

Trapezoidal integration on grid: ∫ f(r) dr = ∫ f(r(x)) * rp(x) dx
"""
function integrate(grid, f::Vector{Float64})
    s = 0.0
    δ = grid.δ
    for i in 1:grid.N-1
        s += 0.5 * δ * (f[i] * grid.rp[i] + f[i+1] * grid.rp[i+1])
    end
    return s
end

# ============================================================================
# Two-way Numerov Solver
# ============================================================================

"""
    compute_g(r, V, E)

Compute g(r) = 2(V - E) for the radial SE.
Note: V should already include the centrifugal term l(l+1)/(2r²) if applicable.
"""
@inline function compute_g(r::Float64, V::Float64, E::Float64)
    return 2.0 * (V - E)
end

"""
    numerov_step(y_curr, y_prev, F_curr, F_prev, F_next, h2)

Single Numerov step for equation y'' = F*y.

Standard Numerov is derived for y'' + f*y = 0, which is y'' = -f*y.
So for y'' = F*y, we use f = -F in the formula:
y_next = [2*y_curr*(1 + 5h²F_curr/12) - y_prev*(1 - h²F_prev/12)] / (1 - h²F_next/12)
"""
@inline function numerov_step(y_curr::Float64, y_prev::Float64,
                               F_curr::Float64, F_prev::Float64, F_next::Float64, h2::Float64)
    # Using f = -F: coefficients become 1-h²f/12 = 1+h²F/12, 1+5h²f/12 = 1-5h²F/12
    # Wait, need to be more careful. Standard formula with f:
    # y[n+1](1 + h²f/12) = 2y[n](1 - 5h²f/12) - y[n-1](1 + h²f/12)
    # With f = -F:
    # y[n+1](1 - h²F/12) = 2y[n](1 + 5h²F/12) - y[n-1](1 - h²F/12)
    c_prev = 1.0 - h2 * F_prev / 12.0
    c_curr = 1.0 + 5.0 * h2 * F_curr / 12.0
    c_next = 1.0 - h2 * F_next / 12.0

    y_next = (2.0 * y_curr * c_curr - y_prev * c_prev) / c_next

    # Overflow protection: clamp to reasonable values
    if abs(y_next) > 1e100
        @warn "Numerov step overflow: clamping |y| > 1e100. Consider reducing r_max or using calculate_safe_rmax()" maxlog=1
        y_next = sign(y_next) * 1e100
    end

    return y_next
end

"""
    calculate_safe_rmax(Z, E_deep)

Calculate a safe r_max that avoids overflow in Numerov integration.
For deeply bound states with energy E, κ = sqrt(-2E), and we need κ*r_max < 300
to avoid extreme underflow/overflow issues.

Also considers that valence orbitals need sufficient radial extent.
"""
function calculate_safe_rmax(Z::Float64, E_deep::Float64=-0.5*Z^2)
    # κ for deepest state
    κ_deep = sqrt(max(1.0, -2.0 * E_deep))

    # Ensure exp(-κr) doesn't completely underflow (κr < 300)
    r_max_deep = 300.0 / κ_deep

    # For valence, need at least r_max = 30 for good coverage
    r_max_valence = 30.0

    # Also limit based on practical considerations
    r_max_upper = 40.0  # No need to go beyond 40 a.u. for most atoms

    return min(r_max_upper, max(r_max_valence, r_max_deep))
end


"""
    numerov_outward!(u, grid, V, l, E, i_max)

Integrate outward from r[1] to r[i_max] using Numerov method.
Works in y-space: y = u / sqrt(rp), with equation y'' = F*y
where F = rp² * g + 1/4.
"""
function numerov_outward!(u::Vector{Float64}, grid, V::Vector{Float64}, l::Int, E::Float64, i_max::Int)
    δ = grid.δ
    h2 = δ * δ

    # Compute F = rp² * g + 1/4 for all points
    F = Vector{Float64}(undef, i_max)
    for i in 1:i_max
        g = compute_g(grid.r[i], V[i], E)
        # Clamp F to prevent numerical instability from very large values
        F_raw = grid.rp2[i] * g + 0.25
        F[i] = clamp(F_raw, -1e6, 1e6)
    end

    # Initial conditions: u ~ r^(l+1) as r→0
    # Transform to y = u/sqrt(rp)
    y = Vector{Float64}(undef, i_max)
    y[1] = grid.r[1]^(l + 1) / sqrt(grid.rp[1])
    y[2] = grid.r[2]^(l + 1) / sqrt(grid.rp[2])

    # Outward Numerov iteration
    for i in 2:i_max-1
        y[i+1] = numerov_step(y[i], y[i-1], F[i], F[i-1], F[i+1], h2)
    end

    # Transform back: u = y * sqrt(rp)
    for i in 1:i_max
        u[i] = y[i] * sqrt(grid.rp[i])
    end
end

"""
    numerov_inward!(u, grid, V, l, E, i_min)

Integrate inward from a suitable starting point to r[i_min] using Numerov method.
Boundary: u ~ exp(-κr) as r→∞ where κ = sqrt(-2E).

Key fix for overflow: Start at r where exp(-κr) is meaningful (κr < 50)
rather than at the grid boundary where it may underflow to zero.
"""
function numerov_inward!(u::Vector{Float64}, grid, V::Vector{Float64}, l::Int, E::Float64, i_min::Int)
    N = grid.N
    δ = grid.δ
    h2 = δ * δ

    # Asymptotic BC: u ~ exp(-κr) where κ = sqrt(-2E)
    κ = sqrt(max(1e-10, -2.0 * E))

    # Find starting point where exp(-κr) is still numerically meaningful
    # Use κr < 50 to ensure exp(-κr) > 1e-22 (well above machine epsilon)
    # This is more conservative than 690 to avoid numerical noise dominating
    max_kr = 50.0
    i_start = i_min + 10  # At least a few points from i_min
    for i in N:-1:i_min+2
        if κ * grid.r[i] < max_kr
            i_start = i
            break
        end
    end

    # Ensure we have enough points to integrate
    if i_start <= i_min + 5
        i_start = min(i_min + 50, N)
    end

    # Compute F for range [i_min, i_start]
    F = Vector{Float64}(undef, N)
    for i in i_min:i_start
        g = compute_g(grid.r[i], V[i], E)
        # Clamp F to prevent numerical instability from very large values
        F_raw = grid.rp2[i] * g + 0.25
        F[i] = clamp(F_raw, -1e6, 1e6)
    end

    y = zeros(N)
    # Use asymptotic form: u ~ exp(-κr), y = u/sqrt(rp)
    y[i_start] = exp(-κ * grid.r[i_start]) / sqrt(grid.rp[i_start])
    if i_start > 1
        y[i_start-1] = exp(-κ * grid.r[i_start-1]) / sqrt(grid.rp[i_start-1])
    end

    # Backward Numerov iteration
    for i in i_start-1:-1:i_min+1
        y[i-1] = numerov_step(y[i], y[i+1], F[i], F[i+1], F[i-1], h2)
    end

    # Transform back
    for i in i_min:i_start
        u[i] = y[i] * sqrt(grid.rp[i])
    end
    # Set remaining points to zero (beyond i_start)
    for i in i_start+1:N
        u[i] = 0.0
    end
end

"""
    find_classical_turning_point(grid, V, E)

Find index of classical turning point where V(r) = E.
For bound states, V goes from -∞ near origin to 0 at ∞.
We search for where V crosses E from below (V[i-1] < E and V[i] >= E).
"""
function find_classical_turning_point(grid, V::Vector{Float64}, E::Float64)
    for i in 2:grid.N
        if V[i] >= E && V[i-1] < E
            return i
        end
    end
    # Fallback for very bound states where turning point is beyond grid
    return div(2 * grid.N, 3)
end

"""
    count_nodes(u, i_start, i_end)

Count zero crossings of u in range [i_start, i_end].
"""
function count_nodes(u::Vector{Float64}, i_start::Int, i_end::Int)
    nodes = 0
    for i in i_start:i_end-1
        if u[i] * u[i+1] < 0
            nodes += 1
        end
    end
    return nodes
end

"""
    log_deriv_mismatch(u_out, u_in, δ, i_match)

Compute logarithmic derivative mismatch Δ = d/dx[ln(u_out)] - d/dx[ln(u_in)]
at the matching point.
"""
function log_deriv_mismatch(u_out::Vector{Float64}, u_in::Vector{Float64}, δ::Float64, i_match::Int)
    # Scale u_in to match u_out at matching point
    scale = u_out[i_match] / u_in[i_match]
    
    # 5-point derivative approximation for better accuracy
    ld_out = (u_out[i_match+1] - u_out[i_match-1]) / (2.0 * δ * u_out[i_match])
    ld_in = scale * (u_in[i_match+1] - u_in[i_match-1]) / (2.0 * δ * u_out[i_match])
    
    return ld_out - ld_in
end

"""
    solve_rse_numerov(grid, V, l, E_guess; target_nodes=-1, E_tol=1e-10, max_iter=100)

Solve radial Schrödinger equation using inward-outward Numerov matching
with node-guided eigenvalue search.

# Algorithm
1. Start from E_guess
2. Use node count to guide energy search (node theorem: nodes increase with E for fixed l)
3. Bracket the eigenvalue by finding E_lo (nodes < target) and E_hi (nodes >= target)
4. Bisect on log-derivative mismatch within the bracket
5. Validate final solution has correct node count

# Arguments
- `grid`: Radial grid (ShiftedExpGrid)
- `V`: Potential V(r) including centrifugal term
- `l`: Angular momentum
- `E_guess`: Initial energy guess (must be negative for bound states)
- `target_nodes`: Expected radial nodes (n-l-1). -1 to skip node check.
- `E_tol`: Energy convergence tolerance
- `max_iter`: Maximum iterations

# Returns
- `E`: Eigenvalue
- `u`: Normalized wavefunction u(r) = r*R(r)
"""
function solve_rse_numerov(grid, V::Vector{Float64}, l::Int, E_guess::Float64;
                           target_nodes::Int=-1, E_tol::Float64=1e-10, max_iter::Int=100)

    N = grid.N
    u_out = zeros(N)
    u_in = zeros(N)
    u = zeros(N)
    δ = grid.δ

    # Function to compute mismatch and nodes for given E
    function mismatch_and_nodes(E::Float64)
        if E >= 0.0
            return 1e10, -1, 0
        end

        i_turn = find_classical_turning_point(grid, V, E)
        if i_turn == 0
            # No turning point - return large mismatch
            return 1e10, -1, 0
        end

        # Ensure minimum matching point for reliable numerics
        i_match = max(100, min(i_turn - 3, N - 10))

        # Extend outward integration to cover more grid points for node counting
        i_out_max = min(i_match + 50, N - 5)
        numerov_outward!(u_out, grid, V, l, E, i_out_max)
        numerov_inward!(u_in, grid, V, l, E, max(i_match - 50, 5))

        # Check for numerical issues
        if !isfinite(u_out[i_match]) || abs(u_out[i_match]) < 1e-100
            return 1e10, -1, i_match
        end
        if !isfinite(u_in[i_match]) || abs(u_in[i_match]) < 1e-100
            return 1e10, -1, i_match
        end

        m = log_deriv_mismatch(u_out, u_in, δ, i_match)

        # Count nodes in outward wavefunction up to matching point
        nodes = count_nodes(u_out, 2, i_match)

        return m, nodes, i_match
    end

    E_final = E_guess

    if target_nodes >= 0
        # Node-guided eigenvalue search
        # Strategy: First find the energy range where nodes = target_nodes,
        # then scan within that range for mismatch sign change, then bisect.

        E = E_guess
        m, nodes, _ = mismatch_and_nodes(E)

        # Step 1: Find the energy range [E_node_lo, E_node_hi] where nodes = target_nodes
        E_node_lo = E
        E_node_hi = E

        step = 0.5
        max_bracket_iter = 50

        if nodes < target_nodes
            # Need higher energy (less negative) to get more nodes
            # First find E where nodes = target_nodes
            for _ in 1:max_bracket_iter
                E_node_hi = min(E_node_hi + step, -0.001)
                _, nodes_hi, _ = mismatch_and_nodes(E_node_hi)
                if nodes_hi >= target_nodes
                    E_node_lo = E_node_hi - step  # Previous energy had fewer nodes
                    break
                end
                step *= 1.5
            end
            # Now narrow to find exact boundary
            for _ in 1:30
                E_mid = 0.5 * (E_node_lo + E_node_hi)
                _, n_mid, _ = mismatch_and_nodes(E_mid)
                if n_mid < target_nodes
                    E_node_lo = E_mid
                else
                    E_node_hi = E_mid
                end
                if abs(E_node_hi - E_node_lo) < 0.001
                    break
                end
            end
            E_node_lo = E_node_hi  # Start of target_nodes region

        elseif nodes > target_nodes
            # Need lower energy (more negative) to get fewer nodes
            for _ in 1:max_bracket_iter
                E_node_lo = E_node_lo - step
                _, nodes_lo, _ = mismatch_and_nodes(E_node_lo)
                if nodes_lo <= target_nodes
                    E_node_hi = E_node_lo + step
                    break
                end
                step *= 1.5
            end
            # Narrow to find exact boundary
            for _ in 1:30
                E_mid = 0.5 * (E_node_lo + E_node_hi)
                _, n_mid, _ = mismatch_and_nodes(E_mid)
                if n_mid > target_nodes
                    E_node_hi = E_mid
                else
                    E_node_lo = E_mid
                end
                if abs(E_node_hi - E_node_lo) < 0.001
                    break
                end
            end
            E_node_hi = E_node_lo  # End of target_nodes region

        else
            # nodes == target_nodes: We're at correct node count but need to find boundaries
            # Search downward (more negative) to find lower boundary
            step = 0.5
            E_test = E
            for _ in 1:max_bracket_iter
                E_test = E_test - step
                _, n_test, _ = mismatch_and_nodes(E_test)
                if n_test < target_nodes
                    # Found lower boundary
                    E_node_lo = E_test
                    break
                end
                E_node_lo = E_test  # Still at target_nodes, update lower bound
                step *= 1.5
            end

            # Search upward (less negative) to find upper boundary
            step = 0.5
            E_test = E
            for _ in 1:max_bracket_iter
                E_test = min(E_test + step, -0.001)
                _, n_test, _ = mismatch_and_nodes(E_test)
                if n_test > target_nodes
                    # Found upper boundary
                    E_node_hi = E_test
                    break
                end
                E_node_hi = E_test  # Still at target_nodes, update upper bound
                step *= 1.5
                if E_test >= -0.001
                    break
                end
            end
        end

        # Special case: ground state might need different handling
        if target_nodes == 0 && E_node_lo == E_node_hi
            # Expand search for ground state
            E_node_lo = E_guess * 2.0  # Go deeper
            E_node_hi = min(-0.001, E_guess * 0.1)  # Go shallower
        end

        # Step 2: Within the target_nodes region, scan for mismatch sign change
        # Use a wide search range to handle screening effects in atoms
        # For valence orbitals, true eigenvalue can be much shallower than hydrogenic guess
        E_search_lo = max(E_node_lo, E_guess * 5.0)  # Go at most 5x deeper than guess
        E_search_hi = min(E_node_hi, -0.001)  # Go all the way to continuum threshold

        # Scan for mismatch sign change
        n_scan = 100  # Use more points for wider search range
        dE = (E_search_hi - E_search_lo) / n_scan
        E_lo = E_search_lo
        m_lo, n_lo, _ = mismatch_and_nodes(E_lo)

        found_bracket = false
        E_hi = E_lo

        for i in 1:n_scan
            E_hi = E_search_lo + i * dE
            m_hi, n_hi, _ = mismatch_and_nodes(E_hi)

            # Check for mismatch sign change within correct node region
            if n_lo == target_nodes && n_hi == target_nodes && m_lo * m_hi < 0
                found_bracket = true
                break
            end

            if n_hi == target_nodes
                E_lo = E_hi
                m_lo = m_hi
                n_lo = n_hi
            end
        end

        # Step 3: Bisect on mismatch
        if found_bracket
            for _ in 1:60
                E_mid = 0.5 * (E_lo + E_hi)
                m_mid, n_mid, _ = mismatch_and_nodes(E_mid)

                if abs(E_hi - E_lo) < E_tol
                    break
                end

                if n_mid != target_nodes
                    # Somehow left the target node region
                    if n_lo == target_nodes
                        E_hi = E_mid
                    else
                        E_lo = E_mid
                    end
                elseif m_mid * m_lo < 0
                    E_hi = E_mid
                    m_hi = m_mid
                else
                    E_lo = E_mid
                    m_lo = m_mid
                end
            end
            E_final = 0.5 * (E_lo + E_hi)
        else
            # No sign change found - use point with smallest |mismatch|
            E_best = E_search_lo
            m_best = abs(m_lo)
            dE = (E_search_hi - E_search_lo) / 100
            for i in 1:100
                E_test = E_search_lo + i * dE
                m_test, n_test, _ = mismatch_and_nodes(E_test)
                if n_test == target_nodes && abs(m_test) < m_best
                    m_best = abs(m_test)
                    E_best = E_test
                end
            end
            E_final = E_best
        end
    else
        # No target nodes - just find nearest eigenvalue by mismatch bisection
        # (original behavior as fallback)
        E_lo = E_guess - 1.0
        E_hi = min(E_guess + 1.0, -0.001)

        m_lo, _, _ = mismatch_and_nodes(E_lo)
        m_hi, _, _ = mismatch_and_nodes(E_hi)

        # Expand if needed
        while m_lo * m_hi > 0 && E_lo > -100.0
            E_lo -= 1.0
            m_lo, _, _ = mismatch_and_nodes(E_lo)
        end

        if m_lo * m_hi < 0
            for _ in 1:60
                E_mid = 0.5 * (E_lo + E_hi)
                m_mid, _, _ = mismatch_and_nodes(E_mid)

                if abs(E_hi - E_lo) < E_tol
                    break
                end

                if m_mid * m_lo < 0
                    E_hi = E_mid
                else
                    E_lo = E_mid
                    m_lo = m_mid
                end
            end
            E_final = 0.5 * (E_lo + E_hi)
        end
    end

    # Final computation with converged E
    i_turn = find_classical_turning_point(grid, V, E_final)
    i_match = max(5, min(i_turn - 3, N - 10))

    numerov_outward!(u_out, grid, V, l, E_final, i_match + 3)
    numerov_inward!(u_in, grid, V, l, E_final, i_match - 3)

    # Match wavefunctions at i_match
    scale = u_out[i_match] / u_in[i_match]

    for i in 1:i_match
        u[i] = u_out[i]
    end
    for i in i_match+1:N
        u[i] = u_in[i] * scale
    end

    # Normalize
    norm_sq = integrate(grid, u .^ 2)
    if norm_sq > 1e-20
        u ./= sqrt(norm_sq)
    end

    # Node check using full matched wavefunction
    if target_nodes >= 0
        nodes = count_nodes(u, 2, N - 1)
        if nodes != target_nodes
            @warn "Node mismatch: expected $target_nodes, found $nodes"
        end
    end

    return E_final, u
end

# ============================================================================
# Poisson Solver
# ============================================================================
"""
    solve_poisson(grid, n)

Solve radial Poisson equation for density n(r).
Returns Hartree potential V_H(r).
"""
function solve_poisson(grid, n::Vector{Float64})
    N = grid.N
    δ = grid.δ
    r = grid.r
    rp = grid.rp
    
    # Integrands with Jacobian
    f1 = r .^ 2 .* n .* rp  # for ∫₀ʳ r'² n dr'
    f2 = r .* n .* rp       # for ∫ᵣ^∞ r' n dr'
    
    int1 = zeros(N)
    int2 = zeros(N)
    
    # Forward cumulative (0 to r)
    for i in 2:N
        int1[i] = int1[i-1] + 0.5 * δ * (f1[i-1] + f1[i])
    end
    
    # Backward cumulative (r to ∞)
    for i in N-1:-1:1
        int2[i] = int2[i+1] + 0.5 * δ * (f2[i] + f2[i+1])
    end
    
    Vh = (4π ./ r) .* int1 .+ 4π .* int2
    return Vh
end

# ============================================================================
# SCF Solver
# ============================================================================
"""
    solve_scf(grid, Z, config; tol=1e-6, max_iter=100, mix_alpha=0.3)

All-electron SCF for an atom.

# Arguments
- `grid`: Radial grid
- `Z`: Nuclear charge
- `config`: Vector of (n, l, occupation)
- `tol`: Density convergence tolerance
- `max_iter`: Max SCF iterations
- `mix_alpha`: Density mixing

# Returns
- `eigenvalues`: Orbital energies
- `orbitals`: Radial wavefunctions u(r)
- `n_dens`: Electron density
- `V_eff`: Effective potential
"""
function solve_scf(grid, Z::Float64, config::Vector{Tuple{Int,Int,Float64}};
                   tol::Float64=1e-6, max_iter::Int=100, mix_alpha::Float64=0.3)

    N = grid.N
    n_orb = length(config)
    N_elec = sum(c[3] for c in config)

    # Compute Slater screening for each orbital
    # This gives better initial guesses for eigenvalues
    function slater_z_eff(n_q, l_q, config, idx)
        # Slater screening rules (simplified)
        # Electrons in same shell: 0.35 (except 1s: 0.30)
        # Electrons in n-1 shell: 0.85
        # Electrons in n-2 or lower: 1.00

        N_inner = 0.0  # Electrons below this shell
        N_same = 0.0   # Electrons in same shell (excluding this one)

        for (j, (n_j, l_j, occ_j)) in enumerate(config)
            if n_j < n_q
                N_inner += occ_j
            elseif n_j == n_q && j != idx
                N_same += occ_j
            end
        end

        # Slater's rule: s = 0.30 for 1s, 0.35 for same shell, 0.85 for n-1, 1.0 for n-2
        if n_q == 1
            s_same = 0.30
        else
            s_same = 0.35
        end

        # Simplified: assume all inner electrons contribute 1.0
        # and same-shell electrons contribute s_same
        sigma = N_inner + s_same * N_same

        return max(1.0, Z - sigma)
    end

    # Initial density from screened hydrogenic guess
    n_dens = zeros(N)
    for (idx, (n_q, l_q, occ)) in enumerate(config)
        Z_eff = slater_z_eff(n_q, l_q, config, idx)
        psi = grid.r.^(l_q + 1) .* exp.(-Z_eff .* grid.r ./ n_q)
        norm_sq = integrate(grid, psi .^ 2)
        psi ./= sqrt(norm_sq)
        n_dens .+= occ .* (psi .^ 2) ./ (4π .* grid.r .^ 2)
    end

    eigenvalues = zeros(n_orb)
    orbitals = [zeros(N) for _ in 1:n_orb]
    V_eff = zeros(N)

    for iter in 1:max_iter
        # Build V_eff = V_nuc + V_H + V_xc
        V_nuc = -Z ./ grid.r
        V_H = solve_poisson(grid, n_dens)
        V_xc = [lda_pz81(ni)[1] for ni in n_dens]
        V_base = V_nuc .+ V_H .+ V_xc

        # Solve each orbital
        new_n_dens = zeros(N)

        for (idx, (n_q, l_q, occ)) in enumerate(config)
            # Add centrifugal term l(l+1)/(2r²) to V_eff for this orbital
            V_eff = V_base .+ l_q * (l_q + 1) ./ (2 .* grid.r .^ 2)

            if iter == 1
                # Use screened hydrogenic guess
                Z_eff = slater_z_eff(n_q, l_q, config, idx)
                E_guess = -0.5 * (Z_eff / n_q)^2
            else
                E_guess = eigenvalues[idx]
            end
            target_nodes = n_q - l_q - 1

            E, u = solve_rse_numerov(grid, V_eff, l_q, E_guess; target_nodes=target_nodes)

            eigenvalues[idx] = E
            orbitals[idx] = u
            new_n_dens .+= occ .* (u .^ 2) ./ (4π .* grid.r .^ 2)
        end

        # Convergence check
        diff = integrate(grid, abs.(new_n_dens .- n_dens) .* 4π .* grid.r .^ 2)

        if diff < tol
            return eigenvalues, orbitals, n_dens, V_eff
        end

        # Mix densities
        n_dens = (1 - mix_alpha) .* n_dens .+ mix_alpha .* new_n_dens
    end

    @warn "SCF did not converge in $max_iter iterations"
    return eigenvalues, orbitals, n_dens, V_eff
end