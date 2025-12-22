module NCPseudoGen

using FiniteDifferences: central_fdm

export ShiftedExpGrid, lda_pz81
export integrate, solve_rse_numerov, solve_poisson, solve_scf
export calculate_safe_rmax

# NCPP exports
export AtomConfig, PseudoOrbital, NormConservingPP
export select_rc, troullier_martins_pswf, invert_schrodinger
export unscreen_potential, construct_kb_projectors
export validate_pseudopotential, generate_ncpp

# Pre-computed finite difference stencils (shared across module)
const _FD_D1 = central_fdm(5, 1; adapt=0)  # 5-point, 1st derivative
const _FD_D2 = central_fdm(5, 2; adapt=0)  # 5-point, 2nd derivative

"""
    fd_derivative(data, i, δ, order=1)

Compute derivative of discrete data using FiniteDifferences.jl 5-point stencils.

# Arguments
- `data`: Vector of sampled values
- `i`: Index at which to compute derivative
- `δ`: Grid spacing (uniform in transformed x-space)
- `order`: Derivative order (1 or 2)

# Returns
Derivative value at index i
"""
function fd_derivative(data::Vector{Float64}, i::Int, δ::Float64, order::Int=1)
    method = order == 1 ? _FD_D1 : _FD_D2
    g = method.grid   # [-2, -1, 0, 1, 2] for 5-point
    c = method.coefs  # stencil coefficients
    return sum(c[k] * data[i + Int(g[k])] for k in eachindex(g)) / δ^order
end

struct ShiftedExpGrid
    N::Int
    Rp::Float64
    r0::Float64         # minimum radius offset
    δ::Float64          # x step size
    r::Vector{Float64}  # physical radius
    rp::Vector{Float64} # dr/dx
    rp2::Vector{Float64} # (dr/dx)^2
end

function ShiftedExpGrid(Rp::Float64, r0::Float64, r_max::Float64, N::Int)
    # x_max: r_max = Rp(exp(x_max) - 1) + r0
    # => x_max = ln( (r_max - r0)/Rp + 1 )
    x_max = log((r_max - r0) / Rp + 1.0)
    δ = x_max / (N - 1)

    x = [j * δ for j in 0:(N-1)]
    r = Rp .* (exp.(x) .- 1.0) .+ r0

    # dr/dx = Rp * exp(x)
    rp = Rp .* exp.(x)
    rp2 = rp .^ 2

    return ShiftedExpGrid(N, Rp, r0, δ, r, rp, rp2)
end

"""
    lda_pz81(n)

Calculate LDA exchange-correlation potential and energy (PZ-81 parametrization).

# Arguments
- `n`: Electron density

# Returns
- `(Vxc, Exc)`: XC potential and energy per electron (Hartree units)
"""
function lda_pz81(n::Float64)
    if n < 1e-15
        return 0.0, 0.0
    end

    # Wigner-Seitz radius
    rs = (3.0 / (4.0 * π * n))^(1/3)

    # Exchange: Vx = -(3/π)^(1/3) * n^(1/3), Ex = (3/4) * Vx
    term = (3.0 / π)^(1/3)
    vx = -term * n^(1/3)
    ex = 0.75 * vx

    # Correlation (PZ-81)
    vc = 0.0
    ec = 0.0
    if rs >= 1.0
        # Low density
        γ, β1, β2 = -0.1423, 1.0529, 0.3334
        denom = 1.0 + β1 * sqrt(rs) + β2 * rs
        ec = γ / denom
        vc = ec * (1.0 + 7/6*β1*sqrt(rs) + 4/3*β2*rs) / denom
    else
        # High density
        A, B, C, D = 0.0311, -0.0480, 0.0020, -0.0116
        ec = A * log(rs) + B + C * rs * log(rs) + D * rs
        vc = (A - C*rs) * log(rs) + (B - A/3) + 2/3*C*rs - D*rs/3
    end

    return vx + vc, ex + ec
end

include("twoway.jl")
include("ncpp.jl")

end
