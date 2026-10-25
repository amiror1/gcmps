#=
lieb_liniger_correlators.jl — correlation functions of the two-species Lieb-Liniger
gas (ga = 0) at fixed chemical potential μ (grand canonical), for several repulsions c:
    (a) g₁(x) = ⟨ψa†(x) ψa(0)⟩ / na     log-log
    (b) g₂(x) = ⟨na(x) na(0)⟩ / na²     linear, from x = 0 (g₂(0) as a point)

With ga = 0 and c acting within each species, the two species are two copies of
the single-species Lieb-Liniger gas, coupled only through the shared bond (the
bond charge is Na - Nb, truncated to the charges of the bond space) and the
regulator on [Ja⁺, Jb⁺]. At fixed μ the density na depends on c, so distances are
plotted as x·na (x in units of the mean spacing) and γ = c/na.

Script flow:
    for each c in `cs` (all on the same bond space; each starts from the previous c's result):
        minimise E_eff = E - μ n (L-BFGS, grand canonical, preconditioned); the first
            c starts from a random state brought to density n_warmup by a short
            fixed-density warm-up; a result in the vacuum (n ≈ 0) is redone from a warm-up
        record na, nb, γ, E, E_eff, g₂(0), ξ, convergence; compute g₁(x), g₂(x)
    plot (a) and (b) side by side

Output in results/lieb_liniger_correlators/: correlators.png, correlators.txt
(one row per c) and states.jls (Serialization).
=#

include("gcmps_functions.jl")

using Plots
using Random
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
global Lovera = 100.
ga = 0.
μ  = 0.5                   # chemical potential (Tonks limit: na = √μ/π ≈ 0.225 per species)
cs = [100.]  # repulsion within each species
# bond space, the same for every c: charges -(m-1)..m-1, degeneracy m - |q|, D = m² (m = 4: D = 16)
m_bond = [4, 5, 6]
strategy = :grand_canonical
gradtol   = 1e-4
maxiter   = 3000
optimizer = LBFGS(50; verbosity = 1, gradtol = gradtol, maxiter = maxiter)
preconditioner = metric_preconditioner()   # natural gradient, see gcmps_functions.jl

# warm-up away from the vacuum J = 0, which is a stationary point of E - μn (see density_vs_D.jl)
N  = 20.
λ_warmup  = 10.
n_vacuum  = 1e-6
warmup_optimizer = LBFGS(50; verbosity = 1, gradtol = 1e-2, maxiter = maxiter)

# plotted ranges in units of the mean spacing 1/na: x·na up to 50 in (a), 5 in (b)
xna_max_a = 50.
xna_max_b = 5.

bond_space(m) = U₁Space([q => m - abs(q) for q in -(m-1):(m-1)]...)

function warmup(x, c)
    x, = optimize(x -> fg(x...; strategy = strategy, Na = N, Nb = N, λ = λ_warmup, ga = ga, μ = μ, c = c),
                  x, warmup_optimizer; precondition = preconditioner)
    return x
end

# Minimise E - μn for repulsion c from x; if that ends in the vacuum, redo it from a
# warm-up. Returns the state, E_eff, the final |∇E_eff|, the number of evaluations,
# and whether it was redone.
function solve_fixed_number(x, c, V)
    minimise(x) = optimize(x -> fg(x...; strategy = strategy, μ = μ, ga = ga, Na=N, Nb=N, c = c, λ = λ_warmup),
                           x, optimizer; precondition = preconditioner)
    x, fx, gx, numfg, hist = minimise(x)
    retried = real(compute_particle_density(x...)) < n_vacuum
    retried && ((x, fx, gx, numfg, hist) = minimise(warmup(random_state(V), c)))
    return x, fx, hist[end, 2], numfg, retried
end

outdir = mkpath(joinpath(@__DIR__, "results", "lieb_liniger_correlators_with_grand_canonical"))
io = open(joinpath(outdir, "correlators.txt"), "w")
println(io, "# Quadratic number penalty,N = $(N), L/a = $(Lovera), gradtol = $(gradtol), maxiter = $(maxiter), cs = $(cs)")
println(io, "# converged = false: maxiter was hit. retried = true: the first attempt fell into the vacuum")
println(io, "# N = target number per species, ⟨Na⟩ = L·na the number reached")
println(io, "# c\tD\tN\t⟨Na⟩\tγ = c/na\tna\tnb\tE\tE_eff\tg₂(0)\tξ/a\t|∇E_eff|\tevaluations\tconverged\tretried")
flush(io)

# density operator of species a, Ja⁺ ⊗ conj(Ja⁺)
density_a(Ja⁺) = @tensor N[-1,-2;-3,-4] := Ja⁺[-1,1,-3]*conj(Ja⁺[-2,1,-4])

# x range left to autoscale: the data end at x = L/2, i.e. x·na = L·na/2, which depends on na
pa = plot(xscale = :log10, yscale = :log10, ylims = (0.01, 1.05),
          xlabel = "x · na", ylabel = "⟨ψa†(x) ψa(0)⟩ / na", legend = :bottomleft, title = "(a)")
pb = plot(xlims = (0, 5), ylims = (0, 1.2), xlabel = "x · na", ylabel = "⟨na(x) na(0)⟩ / na²",
          legend = :bottomright, title = "(b)")


states = Dict{Tuple{Int, Float64}, Any}()
irun = 0
x_prev = nothing
for (m) in m_bond
    println(io, "# Bond space: m = $(m), D = $(m^2)")  
    V = bond_space(m)   
    set_virtual_space!(V)
    x = nothing
    for (ic, c) in enumerate(cs)
        global irun += 1
        global x_prev
        x = x_prev === nothing ? warmup(random_state(V), c) : embed_state(x_prev, V) 
        x, fx, gnorm, numfg, retried = solve_fixed_number(x, c, V)
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

        # grids in units of a, chosen so that x·na covers the plotted range (at most L/2)
        xs_a = exp10.(range(log10(0.2/na), log10(min(xna_max_a/na, Lovera/2)), length = 150))
        xs_b = collect(range(0.01/na, min(xna_max_b/na, Lovera/2), length = 250))
        g1 = map(xi -> abs(dimensionless_two_point_function_a_a_times_L(F, Ja⁺, Jb⁺, xi)), xs_a) ./ na
        g2 = map(xi -> real(dimensionless_density_density_a(F, Ja⁺, Jb⁺, xi)), xs_b) ./ na^2

        col = palette(:tab10)[irun]
        # ⟨N̂a⟩ = L·na: the particle number actually reached (the target is N)
        lbl = "c = $(c), ⟨Na⟩ = $(round(Lovera*na, digits = 2)), γ = $(round(c/na, sigdigits = 3)), D = $(dim(V))" *
            (converged ? "" : " (not conv.)")
        plot!(pa, xs_a .* na, g1, color = col, label = lbl, linewidth = 1.5)
        plot!(pb, xs_b .* na, g2, color = col, label = lbl, linewidth = 1.5)
        scatter!(pb, [0.], [g2_0], color = col, label = "", markersize = 4)   # g₂(0) from ψ†ψ†ψψ

        states[(m, c)] = x
        x_prev = x
        safe_write(f -> serialize(f, states), joinpath(outdir, "states.jls"); io = io)
        # re-drawn after every c, so it always holds everything finished so far
        safe_write(f -> savefig(plot(pa, pb, layout = (1, 2), size = (1100, 420),
                                    left_margin = 5Plots.mm, bottom_margin = 5Plots.mm), f),
                joinpath(outdir, "correlators.png"); io = io)
    end
end

close(io)
