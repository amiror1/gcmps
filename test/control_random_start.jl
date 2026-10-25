#=
Control run: is the D = 6 → D = 16 embedding steering the optimiser?

Runs D = 16, ga = 2.5 twice with the same settings as G-CMPS.jl:
    embedded : start from the D = 6 result, embedded with embed_state
    random   : start from random_state(V16)
and compares the final E_eff and correlation length ξ.

Reading the result:
    random lower E_eff      -> the embedding steers into a worse minimum
    both equal (E_eff, ξ)   -> embedding only slows the start; the
                               non-convergence comes from the penalty conditioning
    random higher E_eff     -> the embedding helps, as intended

ξ (in units of a) is `correlation_length` from gcmps_functions.jl, the same as
in G-CMPS.jl: the gap between the leading eigenvalues of the neutral (q = 0) and
charged (q = ±1) blocks of the transfer-matrix generator.

Run from the project folder, in the Julia REPL:
    include("test/control_random_start.jl")
It uses :fixed_density by default; set  strategy = :grand_canonical  first to
compare the grand-canonical runs instead.
This is a diagnostic, not a unit test: it runs the full D = 16 optimisation
twice, so it takes as long as the D = 16 part of G-CMPS.jl.
=#

using Random
using LinearAlgebra: eigvals

include(joinpath(@__DIR__, "..", "gcmps_functions.jl"))

# same settings as G-CMPS.jl
Random.seed!(1)
# pick the strategy with  strategy = :grand_canonical  before include(...)
strategy = @isdefined(strategy) ? strategy : :fixed_density
λ = 100.
# the G-CMPS.jl run chain up to (and including) the ga = 2.5 run
runs, target = if strategy == :grand_canonical
    [(ga = 0., μ = 0.), (ga = 0., μ = 5.), (ga = 2.5, μ = 5.)], (ga = 2.5, μ = 5.)
else
    [(ga = 0., N = 0.5), (ga = 2.5, N = 0.5)], (ga = 2.5, N = 0.5)
end
optimizer = LBFGS(20; verbosity = 2, gradtol = 1e-4, maxiter = 2000)
global Lovera = 30.

V6  = U₁Space(-1 => 2, 0 => 2, 1 => 2)
V16 = U₁Space(-3=>1, -2=>2, -1 => 3, 0 => 4, 1=>3 , 2=>2, 3=>1)

cost(run) = x -> fg(x...; strategy = strategy, λ = λ, run...)

# 1) D = 6 warm-start chain, exactly as in G-CMPS.jl, to get the state to embed
set_virtual_space!(V6)
x_start = random_state(V6)
x6 = nothing
for run in runs
    x, _ = optimize(cost(run), x_start, optimizer)
    global x_start = kick_J(x)
    run == target && (global x6 = x)
end

# 2) D = 16 from the embedded state and from a random state
set_virtual_space!(V16)
results = Dict()
for (label, x0) in [("embedded", embed_state(x6, V16)), ("random", random_state(V16))]
    x, fx, gx, numfg, normgradhistory = optimize(cost(target), x0, optimizer)
    results[label] = (E_eff = fx, ξ = correlation_length(x...), grad = normgradhistory[end],
                      numfg = numfg, n = real(compute_a_particle_density(x...)))
end

println("\nD = 16, $(target), $(strategy):")
for label in ("embedded", "random")
    r = results[label]
    println("  $(rpad(label, 9)) E_eff = $(round(r.E_eff, digits = 5)),  ξ = $(round(r.ξ, digits = 3)),  ",
            "n = $(round(r.n, digits = 4)),  |∇E_eff| = $(r.grad),  $(r.numfg) evaluations")
end
ΔE = results["random"].E_eff - results["embedded"].E_eff
verdict = min(results["embedded"].n, results["random"].n) < 1e-3 ?
              "a run ended in the vacuum (n ≈ 0): the comparison says nothing about the embedding" :
          abs(ΔE) < 1e-3 ? "same minimum: the embedding only slows the start" :
          ΔE < 0         ? "random start is lower: the embedding steers into a worse minimum" :
                           "random start is higher: the embedding helps"
println("  E_eff(random) - E_eff(embedded) = $(ΔE)  ->  $(verdict)")
