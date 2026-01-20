#!/usr/bin/env julia
# Visualization script for norm-conserving pseudopotential generation
# Creates plots comparing AE vs PS wavefunctions, potentials, and densities

using NCPseudoGen
using CairoMakie
using Printf

# =============================================================================
# Generate NCPP for Al
# =============================================================================

println("Setting up Al atom calculation...")

# Al atom configuration: [Ne] 3s² 3p¹
Z = 13.0
config = [
    (1, 0, 2.0),  # 1s
    (2, 0, 2.0),  # 2s
    (2, 1, 6.0),  # 2p
    (3, 0, 2.0),  # 3s
    (3, 1, 1.0),  # 3p
]

# Create grid (Rp, r0, r_max, N)
# r_max=15 and N=6000 provide good resolution while avoiding overflow
grid = ShiftedExpGrid(0.005, 1e-6, 15.0, 6000)

# Generate NCPP
const pp, ae = generate_ncpp(grid, Z, config)

# =============================================================================
# Plot 1: AE vs PS Wavefunctions
# =============================================================================

println("\nCreating wavefunction comparison plot...")

fig1 = Figure(size=(900, 400))

for (idx, orb) in enumerate(pp.orbitals)
    l_name = ["s","p","d","f"][orb.l+1]
    ax = Axis(fig1[1, idx],
        xlabel = L"r \; \mathrm{(a.u.)}",
        ylabel = L"u(r)",
        title = L"%$(orb.n)%$(l_name) orbital ($r_c = %$(round(orb.rc, digits=2))$)"
    )

    # Find plot range (up to 2*rc or where wavefunction is small)
    r_plot_max = min(2.5 * orb.rc, 10.0)
    i_max = findfirst(r -> r > r_plot_max, grid.r)
    i_max = isnothing(i_max) ? grid.N : i_max

    r = grid.r[1:i_max]
    u_ae = orb.u_ae[1:i_max]
    u_ps = orb.u_ps[1:i_max]

    # Plot
    lines!(ax, r, u_ae, label="AE", color=:blue, linewidth=2)
    lines!(ax, r, u_ps, label="PS", color=:red, linewidth=2, linestyle=:dash)

    # Mark rc
    vlines!(ax, [orb.rc], color=:gray, linestyle=:dot, label=L"r_c")

    axislegend(ax, position=:rt)
    xlims!(ax, 0, r_plot_max)
end

save("wavefunction_comparison.png", fig1, px_per_unit=2)
println("Saved: wavefunction_comparison.png")

# =============================================================================
# Plot 2: Pseudopotentials
# =============================================================================

println("Creating pseudopotential plot...")

fig2 = Figure(size=(800, 500))
ax2 = Axis(fig2[1, 1],
    xlabel = L"r \; \mathrm{(a.u.)}",
    ylabel = L"V(r) \; \mathrm{(Ha)}",
    title = "Ionic Pseudopotentials"
)

# Plot range
r_plot_max = 5.0
i_max = findfirst(r -> r > r_plot_max, grid.r)
i_max = isnothing(i_max) ? grid.N : i_max

r = grid.r[1:i_max]

# Color scheme for different l
colors = [:blue, :red, :green, :orange]
l_names = ["s", "p", "d", "f"]

# Plot V_local
lines!(ax2, r, pp.V_local[1:i_max],
    label=L"V_\mathrm{local} \; (l=1)", color=:black, linewidth=2)

# Plot semi-local potentials
for (l, V_l) in pp.V_nl
    lines!(ax2, r, V_l[1:i_max],
        label=L"V_{%$(l_names[l+1])} \; (l=%$(l))",
        color=colors[l+1], linewidth=1.5, linestyle=:dash)
end

# Reference: -Z_val/r
V_coulomb = -pp.Z_val ./ r
lines!(ax2, r, V_coulomb, label=L"-Z_\mathrm{val}/r", color=:gray, linestyle=:dot)

axislegend(ax2, position=:rb)
xlims!(ax2, 0, r_plot_max)
ylims!(ax2, -10, 2)

save("pseudopotentials.png", fig2, px_per_unit=2)
println("Saved: pseudopotentials.png")

# =============================================================================
# Plot 3: Wavefunction squared (charge density contribution)
# =============================================================================

println("Creating density comparison plot...")

fig3 = Figure(size=(900, 400))

for (idx, orb) in enumerate(pp.orbitals)
    l_name = ["s","p","d","f"][orb.l+1]
    ax = Axis(fig3[1, idx],
        xlabel = L"r \; \mathrm{(a.u.)}",
        ylabel = L"|u(r)|^2",
        title = L"%$(orb.n)%$(l_name)\text{density}"
    )

    local r_max_plot = min(3.0 * orb.rc, 12.0)
    local i_end = findfirst(x -> x > r_max_plot, grid.r)
    i_end = isnothing(i_end) ? grid.N : i_end

    local r_vals = grid.r[1:i_end]
    rho_ae = orb.u_ae[1:i_end].^2
    rho_ps = orb.u_ps[1:i_end].^2

    lines!(ax, r_vals, rho_ae, label="AE", color=:blue, linewidth=2)
    lines!(ax, r_vals, rho_ps, label="PS", color=:red, linewidth=2, linestyle=:dash)

    # Shade the region inside rc
    i_rc = findfirst(x -> x > orb.rc, grid.r)
    i_rc = isnothing(i_rc) ? 1 : i_rc
    band!(ax, r_vals[1:i_rc], zeros(i_rc), rho_ps[1:i_rc], color=(:red, 0.1))

    vlines!(ax, [orb.rc], color=:gray, linestyle=:dot)

    axislegend(ax, position=:rt)
    xlims!(ax, 0, r_max_plot)
end

save("density_comparison.png", fig3, px_per_unit=2)
println("Saved: density_comparison.png")

# =============================================================================
# Plot 4: KB Projector
# =============================================================================

if !isempty(pp.projectors)
    println("Creating KB projector plot...")

    fig4 = Figure(size=(600, 400))
    ax4 = Axis(fig4[1, 1],
        xlabel = L"r \; \mathrm{(a.u.)}",
        ylabel = L"\chi(r)",
        title = "Kleinman-Bylander Projectors"
    )

    r_plot_max = 5.0
    i_max = findfirst(r -> r > r_plot_max, grid.r)
    i_max = isnothing(i_max) ? grid.N : i_max

    r = grid.r[1:i_max]

    for (l, (χ, E_KB)) in pp.projectors
        lines!(ax4, r, χ[1:i_max],
            label=L"l=%$(l) \; (E_\mathrm{KB}=%$(round(E_KB, digits=3)) \, \mathrm{Ha})",
            color=colors[l+1], linewidth=2)
    end

    axislegend(ax4, position=:rt)
    xlims!(ax4, 0, r_plot_max)

    save("kb_projectors.png", fig4, px_per_unit=2)
    println("Saved: kb_projectors.png")
end

# =============================================================================
# Plot 5: Log-derivative at rc (energy scan)
# =============================================================================

println("Creating log-derivative plot...")

fig5 = Figure(size=(600, 400))
ax5 = Axis(fig5[1, 1],
    xlabel = L"E \; \mathrm{(Ha)}",
    ylabel = L"\frac{d}{dr}\ln u(r_c)",
    title = "Log-derivative matching"
)

# For each orbital, scan energy around eigenvalue
for (idx, orb) in enumerate(pp.orbitals)
    E_ref = orb.eigenvalue
    rc = orb.rc
    i_c = findfirst(r -> r >= rc, grid.r)

    # Energy range: ±0.5 Ha around eigenvalue
    E_range = range(E_ref - 0.3, E_ref + 0.3, length=50)

    ld_ae = Float64[]
    ld_ps = Float64[]

    δ = grid.δ

    for E in E_range
        # For AE: log-derivative at rc (approximate from stored wavefunction)
        # This is just at the reference energy, so we use the stored wavefunction
        if E ≈ E_ref
            du_ae = (orb.u_ae[i_c+1] - orb.u_ae[i_c-1]) / (2 * δ * grid.rp[i_c])
            push!(ld_ae, du_ae / orb.u_ae[i_c])

            du_ps = (orb.u_ps[i_c+1] - orb.u_ps[i_c-1]) / (2 * δ * grid.rp[i_c])
            push!(ld_ps, du_ps / orb.u_ps[i_c])
        else
            push!(ld_ae, NaN)
            push!(ld_ps, NaN)
        end
    end

    # Just mark the eigenvalue point
    du_ae = (orb.u_ae[i_c+1] - orb.u_ae[i_c-1]) / (2 * δ * grid.rp[i_c])
    ld_ae_ref = du_ae / orb.u_ae[i_c]

    du_ps = (orb.u_ps[i_c+1] - orb.u_ps[i_c-1]) / (2 * δ * grid.rp[i_c])
    ld_ps_ref = du_ps / orb.u_ps[i_c]

    l_name = ["s","p","d","f"][orb.l+1]
    scatter!(ax5, [E_ref], [ld_ae_ref],
        label=@sprintf("%d%s AE", orb.n, l_name),
        color=colors[idx], marker=:circle, markersize=12)
    scatter!(ax5, [E_ref], [ld_ps_ref],
        label=@sprintf("%d%s PS", orb.n, l_name),
        color=colors[idx], marker=:cross, markersize=12)
end

axislegend(ax5, position=:lt)

save("logderiv_matching.png", fig5, px_per_unit=2)
println("Saved: logderiv_matching.png")

# =============================================================================
# Summary
# =============================================================================

println("\n" * "="^60)
println("NCPP Generation Summary for Al (Z=$Z)")
println("="^60)
@printf("Valence charge: %.1f\n", pp.Z_val)
println("\nOrbital summary:")
for orb in pp.orbitals
    l_name = ["s","p","d","f"][orb.l+1]
    @printf("  %d%s: E = %.4f Ha, rc = %.2f a.u.\n",
        orb.n, l_name, orb.eigenvalue, orb.rc)
end

println("\nKB projectors:")
for (l, (_χ, E_KB)) in pp.projectors
    l_name = ["s","p","d","f"][l+1]
    @printf("  l=%d (%s): E_KB = %.4f Ha\n", l, l_name, E_KB)
end

println("\nPlots saved:")
println("  - wavefunction_comparison.png")
println("  - pseudopotentials.png")
println("  - density_comparison.png")
println("  - kb_projectors.png")
println("  - logderiv_matching.png")
