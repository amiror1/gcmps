#=
Which `weight` should embed_state use?

embed_state puts the new bond states a gap below the old leading TM eigenvalue,
so their weight in the ring trace is `weight` (see its docstring). We want
    (a) the embedded start to be (almost) the same state as the D = 6 one, and
    (b) a gradient that is not negligible in the new directions,
and in the end (c) a lower E_eff after a fixed number of iterations.

For each weight, starting from the D = 6 state at (ga = 2.5, N = 0.5), fixed
density, this prints:
    ΔE_eff      E_eff(embedded start) - E_eff(D = 6 state)                  (a)
    1-fid/L     1 - |⟨ψ6|ψ16⟩|²/(⟨ψ6|ψ6⟩⟨ψ16|ψ16⟩), per unit length a     (a)
                (a gauge-invariant overlap, so it compares states, not tensors)
    |∇ new|     norm of the gradient in the new directions (every entry
                outside the old D = 6 blocks)                               (b)
    |∇ old|     norm of the gradient in the old directions, for scale
    E_eff(k)    E_eff after `budget` L-BFGS iterations                     (c)
plus the same budget from random_state(V16) as a reference.

Run from the project folder, in the Julia REPL:
    include("test/embed_weight_scan.jl")
Diagnostic, not a unit test: the D = 6 chain plus one short D = 16 run per
weight, a few minutes each.
=#

using Random
using LinearAlgebra: eigvals, norm

include(joinpath(@__DIR__, "..", "gcmps_functions.jl"))

# same settings as the fixed-density runs in G-CMPS.jl
Random.seed!(1)
strategy = :fixed_density
λ = 100.
runs   = [(ga = 0., N = 0.5), (ga = 2.5, N = 0.5)]   # G-CMPS.jl chain up to the target run
target = (ga = 2.5, N = 0.5)
global Lovera = 30.

weights = [exp(-40), 1e-12, 1e-6, 1e-3, 1e-2, 1e-1]   # exp(-40) = current default
budget  = 300                                          # L-BFGS iterations per weight

V6  = U₁Space(-1 => 2, 0 => 2, 1 => 2)
V16 = U₁Space(-3=>1, -2=>2, -1 => 3, 0 => 4, 1=>3 , 2=>2, 3=>1)

cost(run) = x -> fg(x...; strategy = strategy, λ = λ, run...)

# log tr(exp(L*T)) = log Σ exp(L λᵢ), summed over the eigenvalues so it never overflows
function log_tr_exp(T)
    ev = reduce(vcat, [eigvals(b) for (_, b) in blocks(T)])
    m = maximum(real, ev)
    return Lovera*m + log(sum(exp.(Lovera .* (ev .- m))))
end

# mixed transfer matrix ⟨ψ_small| ... |ψ_big⟩: ket from x_big, bra from x_small
function mixed_TM(x_big, x_small)
    F, Ja⁺, Jb⁺ = x_big
    f, ja⁺, jb⁺ = x_small
    id_big   = _virtual_identity(domain(F)[1])
    id_small = _virtual_identity(domain(f)[1])
    @tensor T[-1,-2;-3,-4] := F[-1,1,-3]*conj(id_small[-2,1,-4]) + id_big[-1,1,-3]*conj(f[-2,1,-4]) +
                              Ja⁺[-1,1,-3]*conj(ja⁺[-2,1,-4]) + Jb⁺[-1,1,-3]*conj(jb⁺[-2,1,-4])
    return T
end

# 1 - fidelity per unit length (fidelity itself goes like exp(-L·…), so compare per length)
function infidelity_per_length(x_big, x_small)
    TM_big   = compute_TM(x_big...;   id = _virtual_identity(domain(x_big[1])[1]))
    TM_small = compute_TM(x_small...; id = _virtual_identity(domain(x_small[1])[1]))
    log_fid = 2*real(log_tr_exp(mixed_TM(x_big, x_small))) - real(log_tr_exp(TM_big)) - real(log_tr_exp(TM_small))
    return -log_fid/Lovera
end

# split the gradient of an embedded state into old (D = 6 blocks) and new entries
function gradient_split(g, x_small)
    old² = 0.; new² = 0.
    for (gt, t) in zip(g, x_small)
        for (c, b) in blocks(gt)
            b2 = abs2.(b)
            if c in blocksectors(t)
                m, n = size(block(t, c))
                o = sum(b2[1:m, 1:n])
                old² += o; new² += sum(b2) - o
            else
                new² += sum(b2)
            end
        end
    end
    return sqrt(old²), sqrt(new²)
end

# 1) D = 6 state at the target run, exactly as in G-CMPS.jl
set_virtual_space!(V6)
alg6 = LBFGS(20; verbosity = 1, gradtol = 1e-4, maxiter = 2000)
x_start = random_state(V6)
x6 = nothing
for run in runs
    x, _ = optimize(cost(run), x_start, alg6)
    global x_start = kick_J(x)
    run == target && (global x6 = x)
end
E6 = cost(target)(x6)[1]
println("D = 6, $(target): E_eff = $(E6), n = $(real(compute_a_particle_density(x6...)))")

# 2) D = 16: embed with each weight, measure the start, then run `budget` iterations
set_virtual_space!(V16)
alg16 = LBFGS(20; verbosity = 0, gradtol = 1e-4, maxiter = budget)
rows = []
for w in weights
    Random.seed!(10)   # same embedding noise for every weight
    x0 = embed_state(x6, V16; weight = w)
    E0, g0 = cost(target)(x0)
    g_old, g_new = gradient_split(g0, x6)
    x, fx, _ = optimize(cost(target), x0, alg16)
    push!(rows, (w = w, ΔE = E0 - E6, infid = infidelity_per_length(x0, x6),
                 g_old = g_old, g_new = g_new, E_end = fx, n_end = real(compute_a_particle_density(x...))))
end
Random.seed!(10)
_, E_random, _ = optimize(cost(target), random_state(V16), alg16)

fmt(v) = rpad(string(round(v, sigdigits = 3)), 11)
println("\nD = 16, $(target), $(strategy), $(budget) iterations per run:")
println("  ", join(rpad.(["weight", "ΔE_eff", "1-fid/L", "|∇ old|", "|∇ new|", "E_eff($(budget))", "n($(budget))"], 11)))
for r in rows
    println("  ", fmt(r.w), fmt(r.ΔE), fmt(r.infid), fmt(r.g_old), fmt(r.g_new), fmt(r.E_end), fmt(r.n_end))
end
println("  random start:  E_eff($(budget)) = $(round(E_random, sigdigits = 5))")
println("\nPick the largest weight whose ΔE_eff and 1-fid/L are still small, and check it")
println("has a sizeable |∇ new| and ends at or below the random start.")
