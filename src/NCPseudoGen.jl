module NCPseudoGen

struct ShiftedExpGrid
    N::Int
    Rp::Float64
    r0::Float64         # 最小半径（偏移量）
    δ::Float64          # x 步长
    r::Vector{Float64}   # 物理半径
    rp::Vector{Float64}  # dr/dx
    rp2::Vector{Float64} # (dr/dx)^2
end

function ShiftedExpGrid(Rp::Float64, r0::Float64, r_max::Float64, N::Int)
    # 计算 x_max: r_max = Rp(exp(x_max) - 1) + r0
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
Calculate the LDA Exchange-Correlation potential (PZ-81).
Input: n (electron density)
Output: Vxc (Hartree energy units)
"""
function lda_pz81(n::Float64)
    if n < 1e-15 return 0.0, 0.0 end
    
    # 1. Calculate rs (Wigner-Seitz radius)
    rs = (3.0 / (4.0 * π * n))^(1/3)
    
    # --- Exchange Part ---
    # Vx = -(3/π)^(1/3) * n^(1/3)
    # Ex = (3/4) * Vx
    term = (3.0 / π)^(1/3)
    vx = -term * n^(1/3)
    ex = 0.75 * vx
    
    # --- Correlation Part (PZ-81) ---
    vc = 0.0
    ec = 0.0
    if rs >= 1.0
        # Low density parameters
        γ, β1, β2 = -0.1423, 1.0529, 0.3334
        denom = 1.0 + β1 * sqrt(rs) + β2 * rs
        # Ec is the energy per electron, Vc is the potential
        ec = γ / denom
        vc = ec * (1.0 + 7/6*β1*sqrt(rs) + 4/3*β2*rs) / denom
    else
        # High density parameters
        A, B, C, D = 0.0311, -0.0480, 0.0020, -0.0116
        ec = A * log(rs) + B + C * rs * log(rs) + D * rs
        vc = (A - C*rs) * log(rs) + (B - A/3) + 2/3*C*rs - D*rs/3
    end
    
    return vx + vc, ex + ec
end


end
