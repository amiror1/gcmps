# Ground-state energy of the two-species Lieb-Liniger model as a function of the
# coupling c, at fixed mass m, circle length L and particle number N per species
# (N_a = N_b = N).
# Compares the variational U(1) CMPS energy (for several bond dimensions; auxiliary
# space V ⊗ V with R_a on copy 1 and R_b on copy 2, so [R_a, R_b] = 0) with the
# exact Bethe-ansatz energy, the Bogoliubov (small c) approximation and the
# first-order 1/c (fermionization, large c) approximation. Since the species do not
# interact, all reference energies are twice the single-species ones.
# Run with:  julia plot_energy_vs_c.jl   (or include it in the REPL)

include("2_particle_CMPS.jl")
using Random
using Plots

# ---------------------------------------------------------------------------
# Parameters: tweak these
# ---------------------------------------------------------------------------
m  = 1.0u"m^-1"                      # mass (inverse length)
L  = 1.0u"m"                         # circumference of the circle
N  = 2                              # particle number PER SPECIES (N_a = N_b = N)
λ    = 10.0u"m^-1"                 # penalty weight of the augmented Lagrangian (inverse length)
Ntol = 1e-5                          # stop the multiplier updates once |⟨N⟩ - N| < Ntol
# ([R_a, R_b] = 0 holds by construction, so there is no regularity penalty any more)
cs = range(0.1, 50.0; length = 10)  # couplings for the CMPS runs (dimensionless)

# bond dimensions to compare: (D, qmax) = D virtual states in each U(1) charge
# sector q = -qmax, ..., qmax of ONE copy of V, so the bond dimension per copy is
# Dt = D (2qmax + 1). The auxiliary space V ⊗ V has dimension Dt², L*T is Dt⁴ × Dt⁴ and
# exp(L*T) costs O(Dt¹²). One cost + gradient evaluation takes roughly (measured):
#   (1,2): Dt = 5   ~0.1 s      (2,1): Dt = 6   ~0.4 s
#   (3,1): Dt = 9   ~20 s       (2,2): Dt = 10  ~25 s   ← too slow for a scan
Ds = [(1,1), (2,1)]

alg_stage = LBFGS(20; maxiter = 500, gradtol = 1e-3, verbosity = 1)    # multiplier stages: cheap and loose
alg       = LBFGS(20; maxiter = 10000, gradtol = 1e-4, verbosity = 1)  # final run, at the converged multiplier

# ---------------------------------------------------------------------------
# Variational CMPS energies: one scan over c per bond dimension
# ---------------------------------------------------------------------------
# Within a scan, the optimum for one c is used as the starting point for the
# next c ("warm start"), which is usually much faster than a new random state.
E_cmps = Dict{Tuple{Int,Int}, Vector{typeof(1.0u"m^-1")}}()

for (D, qmax) in Ds
    println("\n=== bond dimension: D = $D per charge sector, charges -$qmax … $qmax (per copy of V) ===")
    Random.seed!(1234)                     # reproducible random initial state
    ψ  = rand_CMPS(D, qmax, L)
    Es = typeof(1.0u"m^-1")[]
    for c in cs
        H = LiebLiniger(m, c, L)
        println("c = ", round(c; digits = 3), ":")
        ψ, E, ν, gnorm = find_groundstate(ψ, H, N; λ, Ntol, alg_stage, alg)
        push!(Es, uconvert(u"m^-1", E))
        N_a, N_b = particle_number(ψ)
        println("c = ", round(c; digits = 3), "   E = ", E, "   E_exact = ", energy_exact(H, N),
                "   ⟨N_a⟩ = ", round(N_a; digits = 6), "   ⟨N_b⟩ = ", round(N_b; digits = 6),
                "   ν = ", ν, "   ‖∇‖ = ", gnorm)
    end
    E_cmps[(D, qmax)] = Es
end

# ---------------------------------------------------------------------------
# Exact (Bethe ansatz), Bogoliubov and 1/c energies, on a finer grid of c
# ---------------------------------------------------------------------------
cs_fine = range(first(cs), last(cs); length = 200)
E_exact = [ustrip(u"m^-1", energy_exact(LiebLiniger(m, c, L), N)) for c in cs_fine]
E_bog   = [ustrip(u"m^-1", energy_bogoliubov(LiebLiniger(m, c, L), N)) for c in cs_fine]

# 1/c result 2 E_F (1 - 4/γ), γ = 2mcL/N per species. For γ < 4 it turns negative and
# diverges as c → 0, which would wreck the y-axis, so it is only drawn where γ ≥ 4
# (NaN = not drawn).
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
           title  = "two-species Lieb-Liniger, N_a = N_b = $N, m = $m, L = $L",
           titlefontsize = 10,
           legend = :topleft,
           # fix the y-range to the physical energies: the Bogoliubov curve plunges to
           # large negative values at large c and would otherwise squash the plot
           ylims  = (0, 1.1 * max(maximum(E_exact),
                                  maximum(maximum(ustrip.(u"m^-1", Es)) for Es in values(E_cmps)))))
plot!(plt, cs_fine, E_bog; label = "Bogoliubov", color = :gray, linestyle = :dash, linewidth = 2)
plot!(plt, cs_fine, E_bos; label = "1/c (fermionization)", color = :gray, linestyle = :dot, linewidth = 2)
for ((D, qmax), marker) in zip(Ds, (:circle, :diamond, :square, :utriangle))
    scatter!(plt, cs, ustrip.(u"m^-1", E_cmps[(D, qmax)]);
             label = "CMPS (V⊗V), D = $D per charge, q = -$qmax…$qmax", marker = marker)
end

savefig(plt, "energy_vs_c.png")
println("\nPlot saved to energy_vs_c.png")
display(plt)
