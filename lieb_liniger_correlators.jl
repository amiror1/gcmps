#=
lieb_liniger_correlators.jl — correlation functions of the two-species Lieb-Liniger
gas (ga = 0) at fixed chemical potential μ (grand canonical), for several repulsions c:
    (a) g₁(x) = ⟨ψa†(x) ψa(0)⟩ / na     log-log
    (b) g₂(x) = ⟨na(x) na(0)⟩ / na²     linear, from x = 0 (g₂(0) as a point)

With ga = 0 and c acting within each species, the two species are two copies of
the single-species Lieb-Liniger gas, coupled only through the shared bond (the
bond charge is Na - Nb, truncated to the charges of the bond space) and the
regulator on [Ja⁺, Jb⁺]. At fixed μ the density na depends on c, so γ = c/na. In (a)
x is in units of a, so every curve ends at the same x = L/2 and decays as e^{-x/ξ}
with ξ in units of a; in (b) it is x·na (units of the mean spacing), so the Friedel
oscillations line up.

Script flow:
    for each bond space in `bond_degeneracies` (increasing D):
        1) for each c in `cs`, in parallel threads except on the first bond space:
            start from the same c on the previous bond space (embed_state); on the first
                bond space from the previous c (so one c after the other), the very first
                run from a short warm-up
            minimise E_eff (L-BFGS, `strategy`, preconditioned); a result in the vacuum
                (n ≈ 0) is redone from a warm-up
            every `monitor_frequency` iterations, record the energy terms, density,
                commutator and transfer-matrix sector weights (optimization_monitor)
        2) for each c, one after the other:
            record na, nb, γ, E, E_eff, g₂(0), ξ, convergence; compute g₁(x), g₂(x)
            add them to the figure: one row per D with (a) and (b) side by side, one
                colour per c

Threads: start Julia with one thread per c, e.g. `julia -t 4 lieb_liniger_correlators.jl`
(or set Julia: Num Threads in VS Code); with one thread the c run one after the other.
The bond spaces cannot run in parallel: set_virtual_space! sets globals the functions read.

Output in results/lieb_liniger_correlators/<strategy>/: correlators.png (one row
per D, all c in each), correlators.txt (one row per D and c), states.jls and results.jls
(Serialization, keyed by (D, c); results.jls holds the row of correlators.txt and the curves
as a named tuple); in curves/: g1_<tag>.csv and g2_<tag>.csv, the plotted curves; in
optimization/: progress_<tag>.png and history_<tag>.csv per optimisation, rewritten during
it. <tag> = D<D>_c<c> (and _retry for a redone optimisation). The files have the same
layout as those of cMPSkit_correlators.jl, to compare the two.
=#

include("gcmps_functions.jl")

using Plots
using Random
using DelimitedFiles: writedlm
using Serialization
using LinearAlgebra: BLAS
Random.seed!(1)
BLAS.set_num_threads(1)   # small charge blocks, see G-CMPS.jl

# Writing a file that never kills the run: on Windows the file can be briefly locked
# (OneDrive syncing it, or the image open in a viewer). Retry, then log a warning.
function safe_write(write_file, path; io = stderr, tries = 5, wait = 2.0)
    for k in 1:tries
        try
            write_file(path)
            return true
        catch err
            k == tries && (println(io, "    WARNING: could not save $(basename(path)): $(sprint(showerror, err))"); return false)
            sleep(wait)
        end
    end
end

# parameters
###################
# Ring length L/a. The correlators on a ring are symmetric about L/2, so they are
# computed up to L/2.
global Lovera = 150.
ga = 0.
μ  = 3.
cs = [1.,10.,100.,1000.]  # repulsion within each species

bond_degeneracies = [(3,2,2), (5,4,3,1), (6,4,3,2)]
strategy = :grand_canonical   # :grand_canonical or :number_variance (fixed N per species)
gradtol   = 1e-8
maxiter   = 3000
optimizer = LBFGS(50; verbosity = 1, gradtol = gradtol, maxiter = maxiter)

# warm-up away from the vacuum J = 0, which is a stationary point of E - μn (see density_vs_D.jl)
N  = 20.
λ_warmup  = 10.
n_vacuum  = 1e-6
warmup_optimizer = LBFGS(50; verbosity = 1, gradtol = 1e-2, maxiter = maxiter)

# Progress monitor: every `monitor_frequency` iterations the observables are recorded
# and the progress plot and history of the optimisation are rewritten
monitor_frequency = 50
regulator_weight  = 200.   # weight of the [Ja⁺, Jb⁺] penalty, in the optimisation and in the monitor

# plotted ranges: x/a from x_min_a to L/2 in (a), x·na up to 5 in (b)
x_min_a   = 0.1
xna_max_b = 5.

# bond space with degeneracy d[|q|+1] for charge q
bond_space(d) = U₁Space([q => d[abs(q)+1] for q in -(length(d)-1):(length(d)-1)]...)

function warmup(x, c)
    x, = optimize(x -> fg(x...; strategy = strategy, Na = N, Nb = N, λ = λ_warmup, ga = ga, μ = μ, c = c, regulator = regulator_weight),
                  x, warmup_optimizer; precondition = metric_preconditioner())
    return x
end

# Minimise E - μn for repulsion c from x; if that ends in the vacuum, redo it from a
# warm-up. Returns the state, E_eff, the final |∇E_eff|, the number of evaluations,
# and whether it was redone.
function solve_fixed_number(x, c, V)
    minimise(x, tag) = optimize(x -> fg(x...; strategy = strategy, μ = μ, ga = ga, Na=N, Nb=N, c = c, λ = λ_warmup, regulator = regulator_weight),
                                x, optimizer; precondition = metric_preconditioner(),
                                finalize! = optimization_monitor(tag, c, V))
    tag = "D$(dim(V))_c$(c)"
    x, fx, gx, numfg, hist = minimise(x, tag)
    retried = real(compute_particle_density(x...)) < n_vacuum
    retried && ((x, fx, gx, numfg, hist) = minimise(warmup(random_state(V), c), tag * "_retry"))
    return x, fx, hist[end, 2], numfg, retried
end

outdir = mkpath(joinpath(@__DIR__, "results", "lieb_liniger_correlators", string(strategy)))
optdir = mkpath(joinpath(outdir, "optimization"))   # progress of each optimisation (optimization_monitor)
curvedir = mkpath(joinpath(outdir, "curves"))       # the plotted g₁ and g₂
io = open(joinpath(outdir, "correlators.txt"), "w")
println(io, "# strategy = $(strategy), μ = $(μ), N = $(N), L/a = $(Lovera), gradtol = $(gradtol), maxiter = $(maxiter), cs = $(cs)")
println(io, "# converged = false: maxiter was hit. retried = true: the first attempt fell into the vacuum")
println(io, "# N = target number per species, ⟨Na⟩ = L·na the number reached")
println(io, "# c\tD\tN\t⟨Na⟩\tγ = c/na\tna\tnb\tE\tE_eff\tg₂(0)\tξ/a\t|∇E_eff|\tevaluations\tconverged\tretried")
flush(io)

# density operator of species a, Ja⁺ ⊗ conj(Ja⁺)
density_a(Ja⁺) = @tensor N[-1,-2;-3,-4] := Ja⁺[-1,1,-3]*conj(Ja⁺[-2,1,-4])

# Plots' GR backend is not thread-safe, so the plotting and file writing of the monitors of
# parallel optimisations holds this lock (as in G-CMPS.jl)
const output_lock = ReentrantLock()

# sector weights below this are left out of the histogram of the progress plot (not the history)
weight_floor = 1e-4

# Callback for OptimKit's `optimize(...; finalize! = optimization_monitor(tag, c, V))`, as in
# G-CMPS.jl. It is called once per iteration with the current tensors x, cost f and gradient
# g, and must return them. Every `monitor_frequency` iterations it records the energy terms,
# density, commutator and transfer-matrix sector weights, and rewrites
# optimization/progress_<tag>.png and optimization/history_<tag>.csv.
function optimization_monitor(tag, c, V; frequency = monitor_frequency)
    title = "D = $(dim(V)), c = $(c), " * (strategy == :grand_canonical ? "μ = $(μ)" : "N = $(N), λ = $(λ_warmup)") * " ($(strategy))"
    charges = TM_charges(V)   # sectors of the transfer matrix, q = q_ket - q_bra
    iters = Int[]
    E_effs = Float64[]
    gradnorms = Float64[]
    kinetic = Float64[]
    electric = Float64[]
    contact = Float64[]
    penalty = Float64[]
    total = Float64[]
    nas = Float64[]
    nbs = Float64[]
    commutator = Float64[]
    weights = zeros(0, length(charges))
    step = 0
    function monitor(x, f, g, _)
        step += 1
        if mod(step, frequency) == 0 # only record results for certain iterations, as this would otherwise slow down
            ρ = compute_ring_density(x...)   # shared by all observables below
            parts = energy_parts(x, ga, c, regulator_weight; ρ = ρ)
            na = real(ring_expectation(density_a(x[2]), ρ))
            push!(iters, step)
            push!(E_effs, f)
            push!(gradnorms, sqrt(real(sum(t -> dot(t, t), g))))
            push!(kinetic, parts.kinetic)
            push!(electric, parts.electric)
            push!(contact, parts.contact)
            push!(penalty, parts.penalty)
            push!(total, parts.total)
            push!(nas, na)
            push!(nbs, real(compute_particle_density(x...; ρ = ρ)) - na)
            push!(commutator, max(parts.commutator, eps()))
            weights = vcat(weights, last.(TM_charge_weights(x...))')

            # plotting (GR is not thread-safe) and writing one thread at a time, see output_lock
            lock(output_lock) do
                p1 = plot(iters, kinetic, linestyle = :dash, label = "kinetic")
                plot!(p1, iters, electric, linestyle = :dot, label = "electric")
                plot!(p1, iters, contact, linestyle = :dashdotdot, label = "contact")
                plot!(p1, iters, penalty, linestyle = :dashdot, label = "regulator")
                plot!(p1, iters, total, color = :black, label = "total E")
                plot!(p1, xlabel = "iteration", title = "iteration $step\nenergy density · a² (E_eff = $(round(f; sigdigits = 4)))")
                p2 = plot(iters, nas, label = "na", xlabel = "iteration",
                          title = "|∇E_eff| = $(round(gradnorms[end]; sigdigits = 2))\ndensity per species · a (⟨Na⟩ = $(round(Lovera*nas[end]; digits = 3)), γ = $(round(c/nas[end]; sigdigits = 3)))")
                plot!(p2, iters, nbs, linestyle = :dash, label = "nb")
                strategy == :number_variance && hline!(p2, [N/Lovera], color = :gray, linestyle = :dash, label = "target")
                p3 = plot(iters, commutator, yscale = :log10, color = :black, label = "", xlabel = "iteration",
                          title = "regularity\n⟨[Ja⁺,Jb⁺]†[Ja⁺,Jb⁺]⟩")
                # histogram of the current weights over q, bars from weight_floor up: a weight below
                # it (down to an underflow to 0) gets a bar of zero height
                p4 = bar(charges, max.(weights[end, :], weight_floor), fillto = weight_floor, yscale = :log10,
                         ylims = (weight_floor, 1.), xticks = charges, color = :gray, label = "", xlabel = "q = q_ket - q_bra",
                         title = "transfer-matrix sector weights (iteration $step)\n|tr ℙq exp(L·TM)| (normalised)")
                safe_write(path -> savefig(plot(p1, p2, p3, p4, layout = (2, 2), titlefontsize = 10, legendfontsize = 7, size = (1000, 700),
                                                plot_title = title, plot_titlefontsize = 12,
                                                left_margin = 3Plots.mm, bottom_margin = 3Plots.mm), path),
                           joinpath(optdir, "progress_$tag.png"); io = io)
                table = [permutedims(["iteration", "E_eff", "normgrad", "kinetic", "electric", "contact", "regulator", "E", "na", "nb", "commutator",
                                      ["weight_q=$(q)" for q in charges]...]);
                         iters E_effs gradnorms kinetic electric contact penalty total nas nbs commutator weights]
                safe_write(path -> writedlm(path, table, ','), joinpath(optdir, "history_$tag.csv"); io = io)
            end
        end
        return x, f, g # optimize expects the tensors, cost and gradient back
    end
    return monitor
end

# One figure, one row per bond space: (a) g₁ and (b) g₂, one colour per c (the same in every row)
Ds = [dim(bond_space(d)) for d in bond_degeneracies]
pas = [plot(xscale = :log10, yscale = :log10, xlims = (x_min_a, Lovera/2), ylims = (0.01, 1.05),
            xlabel = "x / a", ylabel = "⟨ψa†(x) ψa(0)⟩ / na", legend = :bottomleft, title = "(a) D = $(D)")
       for D in Ds]
pbs = [plot(xlims = (0, 5), ylims = (0, 1.2), xlabel = "x · na", ylabel = "⟨na(x) na(0)⟩ / na²",
            legend = :bottomright, title = "(b) D = $(D)")
       for D in Ds]
c_colors = palette(:tab10)

states  = Dict{Tuple{Int, Float64}, Any}()
saved_results = Dict{Tuple{Int, Float64}, Any}()   # observables and curves per (D, c), for results.jls
for (iD, d) in enumerate(bond_degeneracies)
    V = bond_space(d)
    println(io, "# Bond space: degeneracies $(d) for q = 0, ±1, ..., D = $(dim(V))")
    set_virtual_space!(V)

    # 1) Optimisations, solve_fixed_number's result per c. Each c starts from the same c on
    # the previous bond space, so they are independent and run in parallel threads; on the
    # first bond space each c starts from the previous c (the first from a warm-up), so
    # there they run one after the other.
    results = Vector{Any}(undef, length(cs))
    t_optimise = @elapsed if iD == 1
        x_start = warmup(random_state(V), cs[1])
        for ic in eachindex(cs)
            results[ic] = solve_fixed_number(x_start, cs[ic], V)
            x_start = results[ic][1]
        end
    else
        D_prev = dim(bond_space(bond_degeneracies[iD-1]))
        Threads.@threads :dynamic for ic in eachindex(cs)
            results[ic] = solve_fixed_number(embed_state(states[(D_prev, cs[ic])], V), cs[ic], V)
        end
    end
    println(io, "# optimising took $(round(t_optimise, digits = 1)) s, ",
            iD == 1 ? "one c after the other" : "in parallel on $(Threads.nthreads()) threads")

    # 2) Observables, table, figure and states.jls, one c after the other
    for (ic, c) in enumerate(cs)
        x, fx, gnorm, numfg, retried = results[ic]
        converged = gnorm < gradtol
        F, Ja⁺, Jb⁺ = x
        n = real(compute_particle_density(F, Ja⁺, Jb⁺))

        ρ  = compute_ring_density(F, Ja⁺, Jb⁺)
        na = real(ring_expectation(density_a(Ja⁺), ρ))
        nb = n - na
        E  = real(compute_a_squared_energy_density(F, Ja⁺, Jb⁺; ga = ga, c = c, ρ = ρ))
        g2_0 = real(compute_a_pair_density(F, Ja⁺, Jb⁺, Ja⁺; ρ = ρ)) / na^2
        ξ  = correlation_length(F, Ja⁺, Jb⁺)
        println(io, join((c, dim(V), N, Lovera*na, c/na, na, nb, E, fx, g2_0, ξ, gnorm, numfg, converged, retried), '\t'))
        flush(io)

        # grids in units of a: the same for every c in (a); in (b) chosen so that x·na
        # covers the plotted range (at most L/2)
        xs_a = exp10.(range(log10(x_min_a), log10(Lovera/2), length = 150))
        xs_b = collect(range(0.01/na, min(xna_max_b/na, Lovera/2), length = 250))
        g1 = map(xi -> abs(dimensionless_two_point_function_a_a_times_L(F, Ja⁺, Jb⁺, xi)), xs_a) ./ na
        g2 = map(xi -> real(dimensionless_density_density_a(F, Ja⁺, Jb⁺, xi)), xs_b) ./ na^2

        col = c_colors[mod1(ic, length(c_colors))]                    # one colour per c
        # ⟨N̂a⟩ = L·na: the particle number actually reached (the target is N)
        lbl = "c = $(c), γ = $(round(c/na, sigdigits = 3)), ⟨Na⟩ = $(round(Lovera*na, digits = 2))" *
            (converged ? "" : " (not conv.)")
        plot!(pas[iD], xs_a, g1, color = col, label = lbl, linewidth = 1.5)
        plot!(pbs[iD], xs_b .* na, g2, color = col, label = lbl, linewidth = 1.5)
        scatter!(pbs[iD], [0.], [g2_0], color = col, label = "", markersize = 4)   # g₂(0) from ψ†ψ†ψψ

        # the curves and the observables, to compare with cMPSkit_correlators.jl without
        # recomputing anything
        tag = "D$(dim(V))_c$(c)"
        safe_write(f -> writedlm(f, [permutedims(["x/a", "g1"]); xs_a g1], ','), joinpath(curvedir, "g1_$tag.csv"); io = io)
        safe_write(f -> writedlm(f, [permutedims(["x/a", "x*na", "g2"]); xs_b xs_b .* na g2], ','), joinpath(curvedir, "g2_$tag.csv"); io = io)
        saved_results[(dim(V), c)] = (; c, D = dim(V), na, nb, γ = c/na, E, E_eff = fx, g2_0, ξ,
                                      gnorm, numfg, converged, retried, xs_a, g1, xs_b, g2)
        safe_write(f -> serialize(f, saved_results), joinpath(outdir, "results.jls"); io = io)

        states[(dim(V), c)] = x
        safe_write(f -> serialize(f, states), joinpath(outdir, "states.jls"); io = io)
        # the figure, re-drawn after every (D, c), so it always holds everything finished so
        # far: the rows of the bond spaces done or in progress
        rows = vcat(([pas[k], pbs[k]] for k in 1:iD)...)
        safe_write(f -> savefig(plot(rows..., layout = (iD, 2), size = (1300, 450*iD), legendfontsize = 7,
                                    plot_title = "$(strategy), μ = $(μ), L/a = $(Lovera)",
                                    left_margin = 5Plots.mm, bottom_margin = 5Plots.mm), f),
                joinpath(outdir, "correlators.png"); io = io)
    end
end

close(io)
