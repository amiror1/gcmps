# Ground-state energy of the Lieb-Liniger model as a function of the coupling c,
# at fixed mass m, circle length L and particle number N.
# Compares the variational CMPS energy (for several bond dimensions) with the
# exact Bethe-ansatz energy, the Bogoliubov (small c) approximation and the
# first-order 1/c (fermionization, large c) approximation.
# Run with:  julia plot_energy_vs_c.jl   (or include it in the REPL)

include("1_particle_CMPS.jl")
using Random
using Plots

# ---------------------------------------------------------------------------
# Parameters: tweak these
# ---------------------------------------------------------------------------
m  = 1.0u"m^-1"                      # mass (inverse length)
L  = 1.0u"m"                         # circumference of the circle
N  = 10                              # particle number
λ  = 1000.0u"m^-1"                   # penalty strength λ(⟨N⟩ - N)² (inverse length)
cs = range(0.1, 250.0; length = 50)   # couplings for the CMPS runs (dimensionless)

Ds = [(2, 2), (3, 3)]                # bond dimensions (D0 even, D1 odd) to compare

alg = LBFGS(20; maxiter = 1000, gradtol = 1e-6, verbosity = 1)   # 1: only warnings

# ---------------------------------------------------------------------------
# Variational CMPS energies: one scan over c per bond dimension
# ---------------------------------------------------------------------------
# Within a scan, the optimum for one c is used as the starting point for the
# next c ("warm start"), which is usually much faster than a new random state.
E_cmps = Dict{Tuple{Int,Int}, Vector{typeof(1.0u"m^-1")}}()

for (D0, D1) in Ds
    println("\n=== bond dimension (D0, D1) = ($D0, $D1) ===")
    Random.seed!(1234)                     # reproducible random initial state
    ψ  = rand_CMPS(D0, D1, L)
    Es = typeof(1.0u"m^-1")[]
    for c in cs
        H = LiebLiniger(m, c, L)
        ψ, E, gnorm = find_groundstate(ψ, H, N; λ = λ, alg = alg)
        push!(Es, uconvert(u"m^-1", E))
        println("c = ", round(c; digits = 3), "   E = ", E, "   E_exact = ", energy_exact(H, N),
                "   ⟨N⟩ = ", round(particle_number(ψ); digits = 4), "   ‖∇‖ = ", gnorm)
    end
    E_cmps[(D0, D1)] = Es
end

# ---------------------------------------------------------------------------
# Exact (Bethe ansatz), Bogoliubov and 1/c energies, on a finer grid of c
# ---------------------------------------------------------------------------
cs_fine = range(first(cs), last(cs); length = 200)
E_exact = [ustrip(u"m^-1", energy_exact(LiebLiniger(m, c, L), N)) for c in cs_fine]
E_bog   = [ustrip(u"m^-1", energy_bogoliubov(LiebLiniger(m, c, L), N)) for c in cs_fine]

# 1/c result E_F (1 - 4/γ), γ = 2mcL/N. For γ < 4 it turns negative and diverges as
# c → 0, which would wreck the y-axis, so it is only drawn where γ ≥ 4 (NaN = not drawn).
γ_min_bos = 4
E_bos = [ustrip(NoUnits, 2 * m * c * L / N) ≥ γ_min_bos ?
         ustrip(u"m^-1", energy_bosonization(LiebLiniger(m, c, L), N)) : NaN for c in cs_fine]
all(isnan, E_bos) && @warn "1/c curve not drawn: γ = 2mcL/N < $γ_min_bos on the whole c range (need c ≥ $(γ_min_bos * N / (2 * ustrip(NoUnits, m * L))))"

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
plt = plot(cs_fine, E_exact; label = "exact (Bethe ansatz)", color = :black, linewidth = 2,
           xlabel = "coupling c",
           ylabel = "ground-state energy E  [m⁻¹]",
           title  = "Lieb-Liniger, N = $N, m = $m, L = $L",
           legend = :topleft,
           # fix the y-range to the physical energies: the Bogoliubov curve plunges to
           # large negative values at large c and would otherwise squash the plot
           ylims  = (0, 1.1 * max(maximum(E_exact),
                                  maximum(maximum(ustrip.(u"m^-1", Es)) for Es in values(E_cmps)))))
plot!(plt, cs_fine, E_bog; label = "Bogoliubov", color = :gray, linestyle = :dash, linewidth = 2)
plot!(plt, cs_fine, E_bos; label = "1/c (fermionization)", color = :gray, linestyle = :dot, linewidth = 2)
for ((D0, D1), marker) in zip(Ds, (:circle, :diamond, :square, :utriangle))
    scatter!(plt, cs, ustrip.(u"m^-1", E_cmps[(D0, D1)]);
             label = "CMPS, D = ($D0, $D1)", marker = marker)
end

savefig(plt, "energy_vs_c.png")
println("\nPlot saved to energy_vs_c.png")
display(plt)
