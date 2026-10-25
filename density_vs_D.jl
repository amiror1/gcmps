#=
density_vs_D.jl — how the density of the free (c = 0) two-species Lieb-Liniger gas
grows with the bond dimension D at fixed chemical potential.

The model has no contact interaction, so ga = 0 is two species of free bosons
(c = 0), plus the regulator that keeps [Ja⁺, Jb⁺] → 0 (see
compute_a_squared_energy_density). For free bosons E_eff = E - μ n has no minimum
at μ > 0 (the density runs away), so at fixed μ the density the cMPS reaches is
set by D alone. This script measures n(D).

Script flow:
    for each μ in `μs`:
        for each bond space, in increasing D (charges -(m-1)..m-1, degeneracy m - |q|, D = m²):
            start: at the smallest D a random state brought to density n_warmup by a
                   short fixed-density warm-up, otherwise the previous D's result embedded
            minimise E_eff = E - μ n (L-BFGS, grand canonical); if that ends in the
                   vacuum (n ≈ 0), redo it from a warm-up on this bond space
            record n = na + nb, na, nb, E, E_eff, ξ, convergence
    plot n versus D, one curve per μ

Output in results/density_vs_D/: density_vs_D.txt (table), density_vs_D.png,
gradhist_mu<μ>_D<D>.png and the optimised states states.jls (Serialization;
load with `using Serialization; deserialize("states.jls")`).
=#

include("gcmps_functions.jl")

using Plots
using Random
using Serialization
using LinearAlgebra: BLAS
Random.seed!(1)
BLAS.set_num_threads(1)   # small charge blocks, see G-CMPS.jl

# parameters
###################
global Lovera = 30.        # ring length L/a
ga = 0.                    # no gauge field: free bosons
c = 20.                    # repulsion strength.
μs = [0.5, 1.0]            # chemical potentials, one D series each
ms = 2:6                   # bond spaces with D = m² = 4, 9, 16, 25, 36

lbfgs_memory = 50
gradtol      = 1e-4
maxiter      = 4000
optimizer = LBFGS(lbfgs_memory; verbosity = 2, gradtol = gradtol, maxiter = maxiter)
# natural-gradient preconditioning with adaptive regularisation (see metric_preconditioner):
# at D = 16, c = 200 it converges where plain L-BFGS crawls
preconditioner = metric_preconditioner()


n_warmup  = 0.5
λ_warmup  = 10.
n_vacuum  = 1e-6
warmup_optimizer = LBFGS(lbfgs_memory; verbosity = 1, gradtol = 1e-2, maxiter = maxiter)

function warmup(x)
    x, = optimize(x -> fg(x...; strategy = :fixed_density, N = n_warmup, λ = λ_warmup, ga = ga, c = c),
                  x, warmup_optimizer; precondition = preconditioner)
    return x
end

# charges -(m-1)..(m-1) with degeneracy m - |q|, total D = m²; each contains the
# previous one, as embed_state requires (m = 4, 5 are the D = 16, 25 of G-CMPS.jl)
bond_space(m) = U₁Space([q => m - abs(q) for q in -(m-1):(m-1)]...)

# density of one species, ⟨J ⊗ conj(J)⟩ (compute_a_particle_density gives na + nb)
function species_density(F, Ja⁺, Jb⁺, J)
    ρ = compute_ring_density(F, Ja⁺, Jb⁺)
    @tensor tmp[-1,-2;-3,-4] := J[-1,1,-3]*conj(J[-2,1,-4])
    return real(ring_expectation(tmp, ρ))
end

outdir = mkpath(joinpath(@__DIR__, "results", "density_vs_D_with_repulsion"))
io = open(joinpath(outdir, "density_vs_D.txt"), "w")
println(io, "# grand canonical, ga = $(ga), L/a = $(Lovera), gradtol = $(gradtol), maxiter = $(maxiter), c =$(c)")
println(io, "# converged = false means maxiter was hit: the density was still moving")
println(io, "# retried = true: the first attempt fell into the vacuum and was redone from a warm-up")
println(io, "# vacuum = true: it fell into the vacuum again, the row is not a finite-density state")
println(io, "# μ\tD\tn\tna\tnb\tE\tE_eff\tξ/a\t|∇E_eff|\tevaluations\tconverged\tretried\tvacuum")
flush(io)

states = Dict{Tuple{Float64,Int}, Any}()
results = Dict(μ => Tuple{Int,Float64}[] for μ in μs)

for μ in μs
    x_prev = nothing
    for m in ms
        V = bond_space(m)
        D = dim(V)
        set_virtual_space!(V)

        minimise_gc(x) = optimize(x -> fg(x...; strategy = :grand_canonical, μ = μ, ga = ga, c=c), x, optimizer;
                                  precondition = preconditioner)
        x_start = x_prev === nothing ? warmup(random_state(V)) : embed_state(x_prev, V)
        x, fx, gx, numfg, hist = minimise_gc(x_start)
        # fell into the vacuum: redo from a warm-up on this bond space
        retried = real(compute_a_particle_density(x...)) < n_vacuum
        if retried
            x, fx, gx, numfg, hist = minimise_gc(warmup(random_state(V)))
        end
        x_prev = x

        F, Ja⁺, Jb⁺ = x
        n  = real(compute_a_particle_density(F, Ja⁺, Jb⁺))
        na = species_density(F, Ja⁺, Jb⁺, Ja⁺)
        nb = species_density(F, Ja⁺, Jb⁺, Jb⁺)
        E  = real(compute_a_squared_energy_density(F, Ja⁺, Jb⁺; ga = ga, c=c))
        ξ  = correlation_length(F, Ja⁺, Jb⁺)
        gnorm = hist[end, 2]
        converged = gnorm < gradtol

        println(io, join((μ, D, n, na, nb, E, fx, ξ, gnorm, numfg, converged, retried, n < n_vacuum), '\t'))
        flush(io)   # finished rows are on disk even if a later D crashes

        push!(results[μ], (D, n))
        states[(μ, D)] = x
        serialize(joinpath(outdir, "states.jls"), states)

        gp = plot(log10.(abs.(hist)), label = ["|E_eff|" "|∇E_eff|"], xlabel = "iteration",
                  ylabel = "log10", title = "μ = $(μ), D = $(D)")
        savefig(gp, joinpath(outdir, "gradhist_mu$(μ)_D$(D).png"))

        # re-drawn after every D, so it always holds everything finished so far
        pl = plot(xlabel = "D", ylabel = "n · a  (na + nb)",
                  title = "Two-species bosons (ga = 0, c = $(c)), L/a = $(Lovera)",
                  titlefontsize = 10, legend = :topleft, marker = :circle)
        for μp in μs
            isempty(results[μp]) || plot!(pl, first.(results[μp]), last.(results[μp]),
                                          label = "μ = $(μp)", marker = :circle)
        end
        savefig(pl, joinpath(outdir, "density_vs_D.png"))
    end
end
close(io)
