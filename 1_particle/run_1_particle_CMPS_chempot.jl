# Optimise a random CMPS for a Lieb-Liniger Hamiltonian at fixed chemical potential μ
# (cost function E - μ⟨N⟩).
# Run with:  julia run_1_particle_CMPS_chempot.jl   (or include it in the REPL)

include("1_particle_CMPS.jl")
using Random

# ---------------------------------------------------------------------------
# Parameters: tweak these
# ---------------------------------------------------------------------------
m  = 1.0u"m^-1"    # mass (inverse length)
c  = 1.0           # coupling (dimensionless)
μ  = 1.0u"m^-1"   # chemical potential (inverse length)
L  = 1.0u"m"       # circumference of the circle

D0, D1 = 10, 10      # bond dimension: D0 even + D1 odd virtual states (total D = D0 + D1)

Random.seed!(1234) # fix the random initial state, for reproducible runs

# ---------------------------------------------------------------------------
# Hamiltonian and random initial state
# ---------------------------------------------------------------------------
H  = LiebLiniger(m, c, L)
ψ0 = rand_CMPS(D0, D1, L)

println("Hamiltonian : m = $m, c = $c, μ = $μ, L = $L")
println("Initial CMPS: virtual space $(virtualspace(ψ0)), bond dimension $(bonddim(ψ0))")
println("Initial energy: ", energy(ψ0, H))

# ---------------------------------------------------------------------------
# Optimise
# ---------------------------------------------------------------------------
alg = LBFGS(20; maxiter = 1000, gradtol = 1e-4, verbosity = 3)   # 3: print f = (E - μ⟨N⟩)·L at every iteration
ψ, E, gnorm = find_groundstate(ψ0, H, μ; alg = alg)

println()
println("Final energy  : ", E)
println("Gradient norm : ", gnorm)
println("Particle number ⟨N⟩: ", particle_number(ψ))

# Compare with the Bethe ansatz at the integer particle number closest to ⟨N⟩
N_BA = round(Int, particle_number(ψ))
println("Bethe ansatz energy (N = $N_BA): ", energy_exact(H, N_BA), "   (relative difference: ", abs((E - energy_exact(H, N_BA)) / energy_exact(H, N_BA)), ")")

# Compare with the 1/c (fermionization) result at the same N; only reliable for γ = 2mcL/N ≫ 1
γ = ustrip(NoUnits, 2 * m * c * L / N_BA)
E_bos = energy_bosonization(H, N_BA)
println("1/c energy          (N = $N_BA): ", E_bos, "   (relative difference: ", abs((E - E_bos) / E_bos), ",  γ = $γ)")
