# Optimise a random CMPS for a Lieb-Liniger Hamiltonian at a desired particle
# number N_desired (cost function E + λ(⟨N⟩ - N_desired)²).
# Run with:  julia run_1_particle_CMPS_Ndes.jl   (or include it in the REPL)

include("1_particle_CMPS.jl")
using Random

# ---------------------------------------------------------------------------
# Parameters: tweak these
# ---------------------------------------------------------------------------
m  = 1.0u"m^-1"       # mass (inverse length)
c  = 1.0              # coupling (dimensionless)
L  = 1.0u"m"          # circumference of the circle

N_desired = 10        # desired number of particles
λ  = 100.0u"m^-1"     # strength of the penalty λ(⟨N⟩ - N_desired)² (inverse length)

D0, D1 = 10, 10       # bond dimension: D0 even + D1 odd virtual states (total D = D0 + D1)

Random.seed!(1234)    # fix the random initial state, for reproducible runs

# ---------------------------------------------------------------------------
# Hamiltonian and random initial state
# ---------------------------------------------------------------------------
H  = LiebLiniger(m, c, L)
ψ0 = rand_CMPS(D0, D1, L)

println("Hamiltonian : m = $m, c = $c, L = $L")
println("Target      : N_desired = $N_desired, penalty λ = $λ")
println("Initial CMPS: virtual space $(virtualspace(ψ0)), bond dimension $(bonddim(ψ0))")
println("Initial energy: ", energy(ψ0, H), "   ⟨N⟩ = ", particle_number(ψ0))

# ---------------------------------------------------------------------------
# Optimise
# ---------------------------------------------------------------------------
alg = LBFGS(20; maxiter = 1000, gradtol = 1e-4, verbosity = 3)   # 3: print f = (E + λ(⟨N⟩ - N_desired)²)·L at every iteration
ψ, E, gnorm = find_groundstate(ψ0, H, N_desired; λ = λ, alg = alg)

println()
println("Final energy        : ", E)
println("Particle number ⟨N⟩ : ", particle_number(ψ), "   (desired: $N_desired)")
println("Gradient norm       : ", gnorm)
println("Bethe ansatz energy : ", energy_exact(H, N_desired), "   (relative difference: ", abs((E - energy_exact(H, N_desired)) / energy_exact(H, N_desired)), ")")

# Compare with the 1/c (fermionization) result; only reliable for γ = 2mcL/N ≫ 1
γ = ustrip(NoUnits, 2 * m * c * L / N_desired)
E_bos = energy_bosonization(H, N_desired)
println("1/c energy          : ", E_bos, "   (relative difference: ", abs((E - E_bos) / E_bos), ",  γ = $γ)")
