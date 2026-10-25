#=
G-CMPS.jl — run script for the gauged cMPS ground-state search.

The model and all functions are in gcmps_functions.jl (see its header).

Script flow:
    for each virtual bond space (bond dimension D = 6, 16 and 25):
        build helper tensors (charge operator, identity) on that space
        first run: random (F, Ja⁺, Jb⁺) at the smallest D, otherwise the
        previous D's first-run result embedded with embed_state; warm start
        each later run from the previous run's result
        for each run in `runs`:
            minimise  E_eff  (L-BFGS), where depending on `strategy`
                :grand_canonical  E_eff = energy - μ * n
                :fixed_density    E_eff = energy + λ * (n - N)²
                :augmented_lagrangian  E_eff = energy - μ (n - N) + λ (n - N)²,
                                  repeated with μ ← μ + 2λ(N - n) until |n - N| < ntol
            plot |⟨ψa†(Δx) ψa(0)⟩| versus Δx
=#

include("gcmps_functions.jl")

using Plots
using Random
using LinearAlgebra: BLAS
Random.seed!(1)
# The charge blocks of the transfer matrix are small (≤ 44×44 at D = 16), so
# multithreaded BLAS only adds overhead: one thread was ~30% faster per evaluation.
BLAS.set_num_threads(1)

# savefig that never kills the run: on Windows the file can be briefly locked
# (OneDrive syncing it, or the image open in a viewer), which fails with
# "Invalid argument". Retry a few times, then log a warning and carry on.
function safe_savefig(p, path; io = stderr, tries = 5, wait = 2.0)
    for k in 1:tries
        try
            savefig(p, path)
            return true
        catch err
            k == tries && (println(io, "    WARNING: could not save $(basename(path)): $(sprint(showerror, err))"); return false)
            sleep(wait)
        end
    end
end

#actual physics :D
###################
# How the particle number is controlled (see compute_dimensionless_effective_energy_density):
#   :grand_canonical -> each run gives (ga, μ)
#   :fixed_density   -> each run gives (ga, N), penalty weight λ below
#   :augmented_lagrangian -> each run gives (ga, N); penalty λ plus a multiplier μ,
#                     updated between optimisations until |n - N| < ntol
strategy = :augmented_lagrangian
λ = 10.
ntol = 1e-3        # augmented Lagrangian: density tolerance |n - N|
max_outer = 10     # augmented Lagrangian: at most this many rounds per run

runs = if strategy == :grand_canonical
    [(ga = 0., μ = 0.), (ga = 0., μ = 1.2), (ga = 2.5, μ = 1.2), (ga = 5., μ = 1.2)]
else
    [(ga = 0., N = 0.5), (ga=0.1, N=0.5), (ga=0.25, N=0.5), (ga=0.5, N=0.5),
    (ga = 1., N = 0.5), (ga = 1.5, N = 0.5)]
end

# Optimiser. L-BFGS remembers the last `lbfgs_memory` steps to estimate the
# curvature, which avoids the zig-zagging of plain gradient descent.
# verbosity: 0 silent, 1 warnings, 2 summary at the end, 3 every iteration
lbfgs_memory = 50
gradtol      = 1e-4    # final tolerance on |∇E_eff|
maxiter      = 3000    # per optimisation (per round for the augmented Lagrangian)
make_optimizer(tol) = LBFGS(lbfgs_memory; verbosity = 1, gradtol = tol, maxiter = maxiter)
optimizer = make_optimizer(gradtol)

# Augmented Lagrangian tolerance schedule: the early rounds are only there to find μ,
# so they stop at a loose gradtol, tightened by `gradtol_shrink` each round down to
# `gradtol`. A run ends only with a round at the final gradtol that also has |n - N| < ntol.
gradtol_start  = 1e-2
gradtol_shrink = 0.1

# One figure for all runs: |two-point function| vs Δx/a on a log x-axis.
# Both axes are dimensionless: distance in units of a, correlator times a

strategy_description = strategy == :fixed_density        ? "fixed density, λ = $(λ)" :
                       strategy == :augmented_lagrangian ? "augmented Lagrangian, λ = $(λ)" : "grand canonical"
pl = plot(xlims = [0.01,10.], ylims = (0., 0.7), xscale =:log10,
          xlabel = "Δx / a",
          ylabel = "|⟨ψa†(Δx) ψa(0)⟩| · a",
          title = "Species-a two-point function\n$(strategy_description), L/a = 30",
          titlefontsize = 11, legendtitle = "D, parameters",
          legend = :outerright, legendfontsize = 7, legendtitlefontsize = 8,
          size = (1300, 600), left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)
# curve styles: one colour per entry of `runs`, one line style per bond space
run_colors   = palette(:tab10)
D_linestyles = [:solid, :dash, :dot]

# Loop over three choices of U(1)-graded virtual (bond) space, written as
# charge => degeneracy:
#   1) charges -1..1,  total bond dimension D = 6
#   2) charges -3..3,  total bond dimension D = 16
#   3) charges -4..4,  total bond dimension D = 25

x_prev = nothing

# All output goes to `results/<strategy>/` next to this script (e.g. results/fixed_density/),
# so runs with different strategies don't overwrite each other: the figures, and the
# run info (convergence, density, E_eff, ξ, norms) as a text file instead of the terminal
outdir = mkpath(joinpath(@__DIR__, "results", String(strategy)))
io = open(joinpath(outdir, "G-CMPS_results.txt"), "w")
println(io, "G-CMPS results: $(strategy_description), L/a = 30, runs = $(runs)")

for (iD, virtual_space) in enumerate([U₁Space(-1 => 2, 0 => 2, 1=>2 ),
     U₁Space(-3=>1, -2=>2, -1 => 3, 0 => 4, 1=>3 , 2=>2, 3=>1),
     U₁Space(-4=>1,-3=>2,-2=>3,-1=>4,0=>5,1=>4,2=>3,3=>2,4=>1) ])
    #some things that remain fixed
    # 1-dim physical spaces carrying the charge of each operator
    global Qa⁺ = U₁Space( 1 => 1)
    global Qb⁺ = U₁Space(-1 => 1)
    global Q_trivial = U₁Space( 0 => 1)

    #virtual_Q and virtual_identity that corresponds to chosen virtual space
    # virtual_Q: diagonal charge operator, each charge-q block set to q * identity
    #            (the electric field operator on the bond)
    global virtual_Q = randn(virtual_space*Q_trivial, virtual_space)
    for (c,b) in blocks(virtual_Q)
        block(virtual_Q, c) .= c.charge*one(b)
    end
    # virtual_identity: identity on the bond, with the trivial physical leg attached
    global virtual_identity = randn(virtual_space*Q_trivial, virtual_space)
    for (c,b) in blocks(virtual_identity)
        block(virtual_identity, c) .= one(b)
    end

    # Ring length in units of a
    global Lovera = 30.

    # Start of the first run: on the smallest bond space a random guess (only
    # symmetry-allowed blocks are filled); on every larger one, the previous bond
    # space's first-run result embedded with embed_state, so the optimiser starts
    # at that state and can only improve on it.
    # Every later run starts from the previous run's optimised tensors (so the order of `runs` matters)
    x_start = if x_prev === nothing
        (randn(ComplexF64, virtual_space*Q_trivial, virtual_space),
         randn(ComplexF64, virtual_space*Qa⁺, virtual_space),
         randn(ComplexF64, virtual_space*Qb⁺, virtual_space))
    else
        embed_state(x_prev, virtual_space)
    end

    # augmented Lagrangian multiplier: starts at 0 on each bond space (μ* depends
    # strongly on D), then each run starts from the previous run's μ
    μ_AL = 0.

    # Scan over `runs`
    for (irun, run) in enumerate(runs)
        ga = run.ga
        # Minimise E_eff (stops once |∇E_eff| < gradtol, or after maxiter iterations)
        if strategy == :augmented_lagrangian
            # outer loop: minimise at fixed μ, update μ from the density reached,
            # warm-start the next minimisation from the result
            x = x_start
            numfg = 0
            # OptimKit's history is a matrix, one row per iteration: [E_eff  |∇E_eff|];
            # the rounds are stacked by rows, `round_starts` marks where each begins
            normgradhistory = zeros(0, 2)
            round_starts = Int[]
            local fx
            tol_k = gradtol_start
            for k in 1:max_outer
                x, fx, gx, numfg_k, hist_k = optimize(x->fg(x...; strategy = strategy, λ = λ, μ = μ_AL, run...), x, make_optimizer(tol_k))
                numfg += numfg_k
                push!(round_starts, size(normgradhistory, 1) + 1)
                normgradhistory = vcat(normgradhistory, hist_k)
                n_k = real(compute_a_particle_density(x...))
                println(io, "D=$(dim(virtual_space)) run $(run), outer $(k): μ = $(μ_AL), gradtol = $(tol_k), n = $(n_k), $(numfg_k) evaluations")
                density_ok = abs(n_k - run.N) < ntol
                density_ok && tol_k <= gradtol && break      # done: on target and fully converged
                if k == max_outer
                    println(io, "    WARNING: stopped after $(max_outer) rounds, |n - N| = $(abs(n_k - run.N)), gradtol = $(tol_k)")
                    break
                end
                # off target: update μ and tighten; on target at a loose tolerance:
                # keep μ and go straight to a final round at the full tolerance
                density_ok || (μ_AL = augmented_lagrangian_update(μ_AL, λ, run.N, n_k))
                tol_k = density_ok ? gradtol : max(gradtol, gradtol_shrink*tol_k)
                tol_k ≈ gradtol && (tol_k = gradtol)   
            end
        else
            x, fx, gx, numfg, normgradhistory = optimize(x->fg(x...; strategy = strategy, λ = λ, run...), x_start, optimizer);
            round_starts = [1]
        end
        # keep the first run's result to embed into the next bond space
        run == runs[1] && (global x_prev = x)
        # J = 0 (the vacuum) is a stationary point of E_eff for every μ, since all
        # terms are quadratic in J. Kick J away from it so the next run can leave it.
        x_start = (x[1],
                   x[2] + 0.1*randn(ComplexF64, virtual_space*Qa⁺, virtual_space),
                   x[3] + 0.1*randn(ComplexF64, virtual_space*Qb⁺, virtual_space))

        # hitting maxiter means the run did NOT converge: don't trust its results
        println(io, "D=$(dim(virtual_space)) run $(run): $(numfg) evaluations, final |∇E_eff| = $(normgradhistory[end, 2])")

        # Optimised tensors
        F   = x[1];
        Ja⁺ = x[2];
        Jb⁺ = x[3];

        # Observables of the converged state
        E_times_L_converged = compute_a_squared_energy_density(F, Ja⁺, Jb⁺; ga = ga)
        PN_times_L_converged = compute_a_particle_density(F, Ja⁺, Jb⁺)

        println(io, "    density n = $(real(PN_times_L_converged))")
        if strategy == :fixed_density
            # the penalty acts as a chemical potential at the optimum
            println(io, "    effective μ = $(2λ*(run.N - real(PN_times_L_converged)))")
        elseif strategy == :augmented_lagrangian
            # chemical potential estimate μ* = dE/dn at the final density
            println(io, "    μ* = $(augmented_lagrangian_update(μ_AL, λ, run.N, PN_times_L_converged))")
        end

        #Correlation length from the transfer-matrix gap: the norm lives in the
        # neutral sector (charge 0), the ψa correlator in the charged sector (±1).
        ξ = correlation_length(F, Ja⁺, Jb⁺)
        println(io, "    E_eff=$(fx)  n=$(real(PN_times_L_converged))  ξ/a=$(ξ)  |∇|=$(normgradhistory[end, 2])")

        # Two-point function ⟨ψa†(Δx) ψa(0)⟩ for Δx/a from 0.01 up to L/a
        xs = collect(0.01 : 0.01 : Lovera)
        ys = map(xi -> dimensionless_two_point_function_a_a_times_L(F, Ja⁺, Jb⁺, xi), xs)

        # label: bond dimension, run parameters, and the density actually reached
        run_label = join(["$(k) = $(v)" for (k, v) in pairs(run)], ", ")
        # colour by run (same colour for the same parameters at every D), line style by D,
        # so the convergence in D at fixed parameters reads off directly
        plot!(pl, xs, norm.(ys), label = "D = $(dim(virtual_space)): $(run_label), n = $(round(real(PN_times_L_converged), digits = 3))",
              color = run_colors[mod1(irun, length(run_colors))],
              linestyle = D_linestyles[mod1(iD, length(D_linestyles))], linewidth = 1.5)
        # re-saved after every run, so it always holds all curves finished so far
        safe_savefig(pl, joinpath(outdir, "two_point_function.png"); io = io)
        # Optimisation history of this run: shows whether the optimiser converged, stalled, or
        # oscillated. Columns of the history: E_eff and |∇E_eff| (abs, since E_eff can be < 0)
        gp = plot(log10.(abs.(normgradhistory)), label = ["|E_eff|" "|∇E_eff|"],
                  xlabel = "iteration", ylabel = "log10",
                  title = "D = $(dim(virtual_space)): $(run)")
        # augmented Lagrangian: dashed line where each new round (μ update) starts
        length(round_starts) > 1 && vline!(gp, round_starts[2:end], color = :gray, linestyle = :dash, label = "μ update")
        safe_savefig(gp, joinpath(outdir, "gradhist_D$(dim(virtual_space))_ga$(ga).png"); io = io)
        println(io, "    norms: F = $(norm(F)), Ja = $(norm(Ja⁺)), Jb = $(norm(Jb⁺))")
        flush(io)   # so the finished runs are on disk even if a later one crashes

    end


end
close(io)


