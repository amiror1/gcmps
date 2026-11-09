#=
G-CMPS.jl — run script for the gauged cMPS ground-state search.

The model and all functions are in gcmps_functions.jl (see its header).

Script flow:
    for each virtual bond space in the list (bond dimension D = 6, 16, 25 and 36):
        set the helper tensors (charge operator, identity) on that space (set_virtual_space!)
        for each run in `runs` (grand canonical: sorted from high to low μ, densest first):
            start: the same run's result on the previous D of this job, or, on the first D of
            the list with `start_from_saved = true`, on the largest saved D in states.jls,
            embedded with embed_state (and kicked if it was the vacuum); these runs are
            independent and are optimised in parallel threads (start Julia with e.g.
            `julia -t 4 G-CMPS.jl`). Runs without such a state start one after the other:
            the first random, the others from the previous run's result, kicked.
            minimise  E_eff  (L-BFGS, preconditioned), where depending on `strategy`
                :grand_canonical  E_eff = energy - μ * n
                :number_variance  E_eff = energy + (λ/L) ⟨(N̂a - Na)²⟩ + (λ/L) ⟨(N̂b - Nb)²⟩
            every `monitor_frequency` iterations, record the energy terms, density,
            commutator and transfer-matrix sector weights (optimization_monitor)
            plot |⟨ψa†(Δx) ψa(0)⟩| versus Δx

Output in results/<strategy>/c<c>/: two_point_function.png (all runs), gradhist_D<D>_<run>.png
(one per run), G-CMPS_results.txt (convergence and observables of every run) and states.jls
(the optimised states, see below); in optimization/: progress_<tag>.png and history_<tag>.csv
per run, rewritten during the optimisation, with <tag> = D<D>_<run parameters>_c<c>.

states.jls is a Dict keyed by (D, run), e.g. (16, (ga = 2.5, μ = 2.2)); each value holds the
tensors x = (F, Ja⁺, Jb⁺) and the run's numbers (c, Lovera, n, E, E_eff, ξ, |∇E_eff|, evaluations).
It is updated after every run and keeps the entries of earlier jobs with the same c. To use a state:

    include("gcmps_functions.jl"); using Serialization
    states = deserialize("results/grand_canonical/c1.0/states.jls")
    s = states[(16, (ga = 2.5, μ = 2.2))]
    set_virtual_space!(domain(s.x[1])[1]); global Lovera = s.Lovera   # the globals the functions read
    dimensionless_two_point_function_a_a_times_L(s.x..., 1.0)
=#

include("gcmps_functions.jl")

# Figures are only saved to files, never shown: no GR window, and the
# top-level plot below ends with `;` so the VS Code plot pane stays empty too
ENV["GKSwstype"] = "nul"
using Plots
using Random
using DelimitedFiles: writedlm
using Serialization
using LinearAlgebra: BLAS
Random.seed!(1)
# The charge blocks of the transfer matrix are small (≤ 44×44 at D = 16), so
# multithreaded BLAS only adds overhead: one thread was ~30% faster per evaluation.
BLAS.set_num_threads(1)

# Writing a file that never kills the run: on Windows the file can be briefly locked
# (OneDrive syncing it, or the image open in a viewer), which fails with
# "Invalid argument". Retry a few times, then log a warning and carry on.
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
safe_savefig(p, path; kwargs...) = safe_write(path -> savefig(p, path), path; kwargs...)

#actual physics :D
###################
# Ring length in units of a
global Lovera = 30.

# How the particle number is controlled (see compute_dimensionless_effective_energy_density):
#   :grand_canonical -> each run gives (ga, μ)
#   :number_variance -> each run gives (ga, Na): Na = target number of particles of each
#                       species on the ring (Nb = Na), penalty weight λ below
strategy = :grand_canonical
λ = 10.
# Contact repulsion within each species, c ψ†ψ†ψψ (Lieb-Liniger; not between a and b),
# the same for every run. γ = c/na: c = 1 at na = 0.25 gives γ = 4.
c = 1.

runs = if strategy == :grand_canonical
    [(ga = 2.5, μ = mu) for mu in [1.9, 2.0, 2.1, 2.2, 2.3, 2.5]]
else
    # Na = 7.5 per species = total density n = 2Na/L = 0.5
    [(ga = 0., Na = 7.5), (ga = 0.1, Na = 7.5), (ga = 0.25, Na = 7.5), (ga = 0.5, Na = 7.5),
     (ga = 1., Na = 7.5), (ga = 1.5, Na = 7.5)]
end
# Grand canonical: densest first (high to low μ), so that each warm start only has to thin
# the gas out, which converges much faster than building it up from the vacuum near the onset
strategy == :grand_canonical && (runs = sort(runs, by = run -> run.μ, rev = true))

# Start the first bond space of the list from saved states (states.jls of this strategy and c):
# each run from its own result on the largest saved bond space that the new one contains,
# e.g. D = 25 when the list holds only D = 36 (or the same D, to continue a run). Runs with
# no usable saved state start as on the smallest bond space. false: always start from scratch.
start_from_saved = true

# Optimiser. L-BFGS remembers the last `lbfgs_memory` steps to estimate the
# curvature, which avoids the zig-zagging of plain gradient descent.
# verbosity: 0 silent, 1 warnings, 2 summary at the end, 3 every iteration
lbfgs_memory = 50
gradtol      = 1e-8    # tolerance on |∇E_eff|
maxiter      = 3000    # per optimisation
optimizer = LBFGS(lbfgs_memory; verbosity = 1, gradtol = gradtol, maxiter = maxiter)
# preconditioner: metric_preconditioner (natural gradient, see gcmps_functions.jl), one per
# optimisation, since it caches the environments of the last state (see optimise_run)

# below this total density a result counts as the vacuum J = 0, a stationary point of E_eff
n_vacuum = 1e-6

# Progress monitor: every `monitor_frequency` iterations the observables are recorded
# and the progress plot and history of the run are rewritten
monitor_frequency = 50.
regulator_weight  = 200.   # weight of the [Ja⁺, Jb⁺] penalty, in the optimisation and in the monitor
# energy_parts and TM_charge_weights: see gcmps_functions.jl

# Callback for OptimKit's `optimize(...; finalize! = optimization_monitor(run, V))`. It is called
# once per iteration with the current tensors x, cost f and gradient g, and must return them.
# Every `monitor_frequency` iterations it records the energy terms, density, commutator and
# transfer-matrix sector weights, and rewrites optimization/progress_<tag>.png and optimization/history_<tag>.csv.
function optimization_monitor(run, V; frequency = monitor_frequency)
    tag = "D$(dim(V))_" * join(["$(k)$(v)" for (k, v) in pairs(run)], "_") * "_c$(c)"
    title = "D = $(dim(V)), " * join(["$(k) = $(v)" for (k, v) in pairs(run)], ", ") * ", c = $(c) ($(strategy_description))"
    charges = TM_charges(V)   # sectors of the transfer matrix, q = q_ket - q_bra
    iters = Int[]
    E_effs = Float64[]
    gradnorms = Float64[]
    kinetic = Float64[]
    electric = Float64[]
    contact = Float64[]
    penalty = Float64[]
    total = Float64[]
    densities = Float64[]
    commutator = Float64[]
    weights = zeros(0, length(charges))
    step = 0
    function monitor(x, f, g, _)
        step += 1
        if mod(step, frequency) == 0 # only record results for certain iterations, as this would otherwise slow down
            ρ = compute_ring_density(x...)   # shared by all observables below
            parts = energy_parts(x, run.ga, c, regulator_weight; ρ = ρ)
            push!(iters, step)
            push!(E_effs, f)
            push!(gradnorms, sqrt(real(sum(t -> dot(t, t), g))))
            push!(kinetic, parts.kinetic)
            push!(electric, parts.electric)
            push!(contact, parts.contact)
            push!(penalty, parts.penalty)
            push!(total, parts.total)
            push!(densities, real(compute_particle_density(x...; ρ = ρ)))
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
                p2 = plot(iters, densities ./ 2, label = "na = nb", xlabel = "iteration",
                          title = "|∇E_eff| = $(round(gradnorms[end]; sigdigits = 2))\ndensity per species · a (⟨Na⟩ = $(round(Lovera*densities[end]/2; digits = 3)))")
                haskey(run, :Na) && hline!(p2, [run.Na/Lovera], color = :gray, linestyle = :dash, label = "target")
                p3 = plot(iters, commutator, yscale = :log10, color = :black, label = "", xlabel = "iteration",
                          title = "regularity\n⟨[Ja⁺,Jb⁺]†[Ja⁺,Jb⁺]⟩")
                # weights below 1e-200 (down to an underflow to 0) are left out of the plot, not the
                # history: a log axis reaching 1e-300 breaks GR and the whole panel stays empty
                p4 = plot(iters, map(w -> w > 1e-200 ? w : NaN, weights), yscale = :log10, label = permutedims(["q = $(q)" for q in charges]),
                          xlabel = "iteration", title = "transfer-matrix sector weights\n|tr ℙq exp(L·TM)| (normalised)")
                safe_savefig(plot(p1, p2, p3, p4, layout = (2, 2), titlefontsize = 10, legendfontsize = 7, size = (1000, 700),
                                  plot_title = title, plot_titlefontsize = 12,
                                  left_margin = 3Plots.mm, bottom_margin = 3Plots.mm),
                             joinpath(optdir, "progress_$tag.png"); io = io)
                table = [permutedims(["iteration", "E_eff", "normgrad", "kinetic", "electric", "contact", "regulator", "E", "n", "commutator",
                                      ["weight_q=$(q)" for q in charges]...]);
                         iters E_effs gradnorms kinetic electric contact penalty total densities commutator weights]
                safe_write(path -> writedlm(path, table, ','), joinpath(optdir, "history_$tag.csv"); io = io)
            end
        end
        return x, f, g # optimize expects the tensors, cost and gradient back
    end
    return monitor
end

# Parallel runs: on the larger bond spaces the runs are independent and are optimised in
# parallel threads (start Julia with threads, e.g. `julia -t 4 G-CMPS.jl`, or set Julia: Num
# Threads in VS Code; one thread runs them one after the other). Plots' GR backend is not
# thread-safe, so all plotting and file writing during the optimisations holds this lock.
const output_lock = ReentrantLock()

# Minimise E_eff for `run` on bond space V from x_start (stops once |∇E_eff| < gradtol, or after
# maxiter iterations). Each call gets its own preconditioner (it caches the environments of the
# last state) and its own monitor, so calls can run in parallel threads.
optimise_run(run, V, x_start) = optimize(x->fg(x...; strategy = strategy, λ = λ, c = c, regulator = regulator_weight, run...), x_start, optimizer;
                                         precondition = metric_preconditioner(),
                                         finalize! = optimization_monitor(run, V))

# One figure for all runs: |two-point function| vs Δx/a on a log x-axis.
# Both axes are dimensionless: distance in units of a, correlator times a
strategy_description = strategy == :number_variance ? "number variance, λ = $(λ)" : "grand canonical"
# The y range follows the largest curve so far (it starts at na, which grows with μ and D)
pl = plot(xlims = [0.01,10.], xscale =:log10,
          xlabel = "Δx / a",
          ylabel = "|⟨ψa†(Δx) ψa(0)⟩| · a",
          title = "Species-a two-point function\n$(strategy_description), L/a = $(Lovera), c = $(c)",
          titlefontsize = 11, legendtitle = "D, parameters",
          legend = :outerright, legendfontsize = 7, legendtitlefontsize = 8,
          size = (1300, 600), left_margin = 5Plots.mm, bottom_margin = 5Plots.mm);
# curve styles: one colour per entry of `runs`, one line style per bond space
run_colors   = palette(:tab10)
D_linestyles = [:solid, :dash, :dot]

# All output goes to `results/<strategy>/` next to this script (e.g. results/number_variance/),
# so runs with different strategies don't overwrite each other: the figures, and the
# run info (convergence, density, E_eff, ξ, norms) as a text file instead of the terminal
outdir = mkpath(joinpath(@__DIR__, "results", String(strategy), "c$(c)"))
optdir = mkpath(joinpath(outdir, "optimization"))   # progress of each optimisation (optimization_monitor)
io = open(joinpath(outdir, "G-CMPS_results.txt"), "w")
println(io, "G-CMPS results: $(strategy_description), L/a = $(Lovera), c = $(c), runs = $(runs), threads = $(Threads.nthreads())")

# Optimised states, keyed by (D, run): the entries of earlier jobs with the same c are kept
# (a rerun of the same D and run replaces its entry)
states_path = joinpath(outdir, "states.jls")
saved_states = Dict{Any, Any}()
if isfile(states_path)
    try
        merge!(saved_states, deserialize(states_path))
    catch err
        println(io, "WARNING: could not read the existing $(basename(states_path)), starting a new one: $(sprint(showerror, err))")
    end
end

# V_big contains V_small: every charge of V_small, with at least the same degeneracy (as embed_state requires)
contains_space(V_big, V_small) = all(s -> dim(V_big, s) >= dim(V_small, s), sectors(V_small))

# Saved start for `run` on bond space V: its entry in states.jls (same L/a) on the largest saved
# bond space with D ≤ dim(V) that V contains, or nothing
function saved_start(run, V)
    entries = [entry for ((D, r), entry) in saved_states
               if r == run && entry.Lovera == Lovera && D <= dim(V) && contains_space(V, domain(entry.x[1])[1])]
    return isempty(entries) ? nothing : entries[argmax([entry.D for entry in entries])]
end

# Loop over three choices of U(1)-graded virtual (bond) space, written as
# charge => degeneracy:
#   1) charges -1..1,  total bond dimension D = 6
#   2) charges -3..3,  total bond dimension D = 16
#   3) charges -4..4,  total bond dimension D = 25
#   4) charges -5..5,  total bond dimension D = 36
prev_states = nothing      # optimised (F, Ja⁺, Jb⁺) of every run on the previous bond space
prev_densities = nothing   # and their densities n, to recognise the vacuum
two_point_max = 0.   # largest |⟨ψa†(Δx) ψa(0)⟩| plotted so far, sets the y range of `pl`

for (iD, virtual_space) in enumerate([
    #  U₁Space(-1 => 2, 0 => 2, 1=>2 ),
    #  U₁Space(-3=>1, -2=>2, -1 => 3, 0 => 4, 1=>3 , 2=>2, 3=>1),
    #  U₁Space(-4=>1,-3=>2,-2=>3,-1=>4,0=>5,1=>4,2=>3,3=>2,4=>1),
     U₁Space(-5=>1,-4=>2,-3=>3,-2=>4,-1=>5,0=>6,1=>5,2=>4,3=>3,4=>2, 5=>1)
      ])
    # charge spaces, charge (electric-field) operator and identity on this bond space
    set_virtual_space!(virtual_space)

    states = Vector{Any}(undef, length(runs))   # results of this bond space, for the next one
    densities = zeros(length(runs))
    optimised = Vector{Any}(undef, length(runs))   # optimize's (x, E_eff, ∇E_eff, evaluations, history) per run

    # 1) Minimise E_eff for every run. J = 0 (the vacuum) is a stationary point of E_eff for
    # every μ, since all terms are quadratic in J, so a start from the vacuum is kicked away from it.
    # Where each run starts: (state on a smaller or equal bond space, its density, description), or
    # nothing. After the first bond space, the same run's result on the previous one; on the first,
    # its saved result (start_from_saved).
    sources = map(eachindex(runs)) do irun
        if prev_states !== nothing
            (prev_states[irun], prev_densities[irun], "D=$(dim(domain(prev_states[irun][1])[1])) result of this job")
        elseif start_from_saved && (entry = saved_start(runs[irun], virtual_space)) !== nothing
            (entry.x, entry.n, "saved D=$(entry.D) state")
        else
            nothing
        end
    end
    from_source = [irun for irun in eachindex(runs) if sources[irun] !== nothing]
    t_optimise = @elapsed begin
        # runs with a state to start from: embedded with embed_state (so the optimiser starts at
        # that state and can only improve on it). They are independent, so they are optimised
        # in parallel threads.
        Threads.@threads :dynamic for irun in from_source
            x_small, n_small, _ = sources[irun]
            x_embedded = embed_state(x_small, virtual_space)
            x_start = n_small < n_vacuum ? kick_J(x_embedded) : x_embedded
            optimised[irun] = optimise_run(runs[irun], virtual_space, x_start)
        end
        # the others: a random guess (only symmetry-allowed blocks are filled) for the first run,
        # the previous run's result for the others (so the order of `runs` matters), hence one
        # run after the other
        for irun in eachindex(runs)
            sources[irun] === nothing || continue
            x_start = irun == 1 ? random_state(virtual_space) : kick_J(optimised[irun - 1][1])
            optimised[irun] = optimise_run(runs[irun], virtual_space, x_start)
        end
    end

    println(io, "D=$(dim(virtual_space)): optimising all runs took $(round(t_optimise, digits = 1)) s ",
            "($(length(from_source)) from a previous state, in parallel on $(Threads.nthreads()) threads; ",
            "$(length(runs) - length(from_source)) from scratch, one after the other)")

    # 2) Observables, figures and saving, one run after the other
    t_observables = @elapsed for (irun, run) in enumerate(runs)
        ga = run.ga
        x, fx, gx, numfg, normgradhistory = optimised[irun]
        # hitting maxiter means the run did NOT converge: don't trust its results
        println(io, "D=$(dim(virtual_space)) run $(run): $(numfg) evaluations, final |∇E_eff| = $(normgradhistory[end, 2])")
        println(io, "    started from: ", sources[irun] === nothing ? (irun == 1 ? "random state" : "previous run (kicked)") : sources[irun][3],
                sources[irun] !== nothing && sources[irun][2] < n_vacuum ? " (vacuum, kicked)" : "")

        # Optimised tensors
        F   = x[1];
        Ja⁺ = x[2];
        Jb⁺ = x[3];

        # Observables of the converged state; n = na + nb is the total density × a, and
        # the ansatz forces na = nb (the bond charge Na - Nb returns to itself around the ring)
        n = real(compute_particle_density(F, Ja⁺, Jb⁺))
        states[irun] = x   # start of the same run on the next bond space
        densities[irun] = n
        E = real(compute_a_squared_energy_density(F, Ja⁺, Jb⁺; ga = ga, c = c))
        println(io, "    density n = $(n), na = nb = $(n/2), ⟨Na⟩ = L·na = $(Lovera*n/2)",
                strategy == :number_variance ? " (target Na = $(run.Na))" : "")
        n < n_vacuum && println(io, "    WARNING: the state is in the vacuum (n < $(n_vacuum))")

        #Correlation length from the transfer-matrix gap: the norm lives in the
        # neutral sector (charge 0), the ψa correlator in the charged sector (±1).
        ξ = correlation_length(F, Ja⁺, Jb⁺)
        println(io, "    E=$(E)  E_eff=$(fx)  ξ/a=$(ξ)  |∇|=$(normgradhistory[end, 2])")

        # save the state, so it can be used later without re-optimising (see the header)
        saved_states[(dim(virtual_space), run)] = (; x, run, D = dim(virtual_space), c, Lovera, strategy, n, E, E_eff = fx, ξ,
                                                     normgrad = normgradhistory[end, 2], evaluations = numfg)
        safe_write(path -> serialize(path, saved_states), states_path; io = io)

        # Two-point function ⟨ψa†(Δx) ψa(0)⟩ for Δx/a from 0.01 to 10, log-spaced like the plot
        # (all distances in one call, which computes what does not depend on Δx only once)
        xs = 10 .^ range(-2, 1, length = 200)
        ys = dimensionless_two_point_function_a_a_times_L(F, Ja⁺, Jb⁺, xs)

        # label: bond dimension, run parameters, and the density actually reached
        run_label = join(["$(k) = $(v)" for (k, v) in pairs(run)], ", ")
        # colour by run (same colour for the same parameters at every D), line style by D,
        # so the convergence in D at fixed parameters reads off directly
        plot!(pl, xs, norm.(ys), label = "D = $(dim(virtual_space)): $(run_label), n = $(round(n, digits = 3))",
              color = run_colors[mod1(irun, length(run_colors))],
              linestyle = D_linestyles[mod1(iD, length(D_linestyles))], linewidth = 1.5)
        global two_point_max = max(two_point_max, maximum(norm, ys))
        plot!(pl, ylims = (0, max(1.05*two_point_max, 0.1)))   # at least 0.1, so vacuum-only plots stay valid
        # re-saved after every run, so it always holds all curves finished so far
        safe_savefig(pl, joinpath(outdir, "two_point_function.png"); io = io)
        # Optimisation history of this run: shows whether the optimiser converged, stalled, or
        # oscillated. Columns of the history: E_eff and |∇E_eff| (abs, since E_eff can be < 0)
        gp = plot(log10.(abs.(normgradhistory)), label = ["|E_eff|" "|∇E_eff|"],
                  xlabel = "iteration", ylabel = "log10",
                  title = "D = $(dim(virtual_space)): $(run)")
        # named by all run parameters: with only ga, the runs of a μ scan would overwrite each other
        safe_savefig(gp, joinpath(outdir, "gradhist_D$(dim(virtual_space))_" * join(["$(k)$(v)" for (k, v) in pairs(run)], "_") * ".png"); io = io)
        println(io, "    norms: F = $(norm(F)), Ja = $(norm(Ja⁺)), Jb = $(norm(Jb⁺))")
        flush(io)   # so the finished runs are on disk even if a later one crashes
    end
    println(io, "D=$(dim(virtual_space)): observables, figures and saving took $(round(t_observables, digits = 1)) s")
    flush(io)
    # this bond space's results start the same runs on the next one
    global prev_states = states
    global prev_densities = densities
end
close(io)
