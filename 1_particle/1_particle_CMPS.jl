#A very simple 1 particle CMPS code in TensorKit language.
#Goals :

#  GS energy --> Compare to Bethe Ansatz
#  GS energy --> Compare to small c, large N, large L perturbation theory
#  GS energy --> Compare to large c bosonzatin results in perturbation theory

#  two point function  --> Compare to Bethe Ansatz
#  two point function  --> Compare to small c, large N, large L perturbation theory
#  two point function --> Compare to large c bosonzatin results in perturbation theory

using TensorKit
using OptimKit
using LinearAlgebra
using Zygote
using Unitful
using UnitfulChainRules

# ---
# Lieb Liniger Hamiltonian type
# ---
#=
Lieb-Liniger Hamiltonian on a circle of length L (PBC), in units ħ = 1 and with
mass in inverse length:

    H = ∫₀ᴸ dx [ 1/(2m) ∂ₓψ†(x) ∂ₓψ(x) + c ψ†(x)ψ†(x)ψ(x)ψ(x) ]

Units, with ψ(x) of dimension length^(-1/2):
  * m : inverse length (e.g. 1u"m^-1")
  * L : length         (e.g. 10u"m")
  * c : dimensionless. Both terms then have units of energy/length = length⁻²,
        and the usual dimensionless Lieb-Liniger parameter is γ = 2mc/n,
        with n the particle density.

The chemical potential μ is not part of H: it is passed to find_groundstate.
=#
struct LiebLiniger
    m::Unitful.Wavenumber   # mass, in units of inverse length
    c::Real                 # coupling, dimensionless
    L::Unitful.Length       # system size (circumference of the circle)
end

#=
Exact ground-state energy with N particles from the Bethe ansatz (repulsive, c ≥ 0).

In first quantisation H = Σᵢ -∂ᵢ²/(2m) + 2c Σ_{i<j} δ(xᵢ - xⱼ)  (the factor 2 because
ψ†ψ†ψψ counts every pair twice). Multiplying by 2m gives the textbook form
-Σᵢ ∂ᵢ² + 2c' Σ_{i<j} δ(xᵢ - xⱼ) with c' = 2mc, whose eigenstates have quasi-momenta kⱼ
solving the Bethe equations (logarithmic form)

    kⱼ L + Σₗ 2 arctan((kⱼ - kₗ)/c') = 2π Iⱼ,   Iⱼ = -(N-1)/2, ..., (N-1)/2  (ground state)

with energy E = Σⱼ kⱼ² / (2m). In terms of the dimensionless xⱼ = kⱼ L and g = c'L = 2mcL:

    xⱼ + Σₗ 2 arctan((xⱼ - xₗ)/g) = 2π Iⱼ,    E = Σⱼ xⱼ² / (2m L²)

These are solved with Newton's method, starting from the hard-core (g → ∞)
solution xⱼ = 2π Iⱼ. Returns a Unitful energy in units of inverse length.
=#
function energy_exact(H::LiebLiniger, N::Integer; tol = 1e-14, maxiter = 1000)
    N ≥ 0  || throw(ArgumentError("N must be non-negative, got $N"))
    H.c ≥ 0 || throw(ArgumentError("only the repulsive case c ≥ 0 is implemented, got c = $(H.c)"))
    L, m = H.L, H.m
    (N ≤ 1 || H.c == 0) && return 0.0 / (m * L^2)   # no interactions: all particles in k = 0

    g = ustrip(NoUnits, 2 * m * H.c * L)   # dimensionless coupling c'L
    I = [j - (N + 1) / 2 for j in 1:N]     # ground-state quantum numbers
    x = 2π .* I                            # initial guess: hard-core bosons

    for _ in 1:maxiter
        # residual of the Bethe equations and its Jacobian
        res = [x[j] + sum(2atan((x[j] - x[l]) / g) for l in 1:N) - 2π * I[j] for j in 1:N]
        J = [j == l ? 1 + sum(2g / (g^2 + (x[j] - x[k])^2) for k in 1:N if k != j) :
                      -2g / (g^2 + (x[j] - x[l])^2) for j in 1:N, l in 1:N]
        dx = J \ res
        x .-= dx
        maximum(abs, dx) < tol * max(1, maximum(abs, x)) && return sum(abs2, x) / (2 * m * L^2)
    end
    error("Bethe equations did not converge in $maxiter Newton steps (N = $N, 2mcL = $g)")
end

#=
Ground-state energy with N particles in the Bogoliubov approximation (c ≥ 0),
accurate for weak coupling γ = 2mc/n ≪ 1.

In the textbook form -Σᵢ ∂ᵢ² + 2c' Σ_{i<j} δ(xᵢ - xⱼ) with c' = 2mc, density n = N/L
and momenta k = 2πj/L on the ring (see e.g. arXiv:1502.03291, arXiv:1609.08865):

    E_Bog = c' N(N-1)/L + Σ_{j≠0} ½ [ √(k⁴ + 4c'n k²) - k² - 2c'n ]

and our energy is E_Bog/(2m). In terms of x = kL = 2πj and g = c'L = 2mcL:

    E = [ g N(N-1) + Σ_{j≠0} ½ (√(x⁴ + 4gN x²) - x² - 2gN) ] / (2m L²)

The summand is evaluated in the equivalent, numerically stable form
-2g²N² / (√(x⁴ + 4gN x²) + x² + 2gN), which behaves as -g²N²/x² for large x.
The sum is done exactly up to |j| = jmax and the tail beyond is added analytically.
Returns a Unitful energy in units of inverse length.
=#
function energy_bogoliubov(H::LiebLiniger, N::Integer; jmax = 10^5)
    N ≥ 0  || throw(ArgumentError("N must be non-negative, got $N"))
    H.c ≥ 0 || throw(ArgumentError("only the repulsive case c ≥ 0 is implemented, got c = $(H.c)"))
    L, m = H.L, H.m
    N ≤ 1 && return 0.0 / (m * L^2)        # no pairs: no interaction energy (the formula would give a spurious < 0 value)
    g = ustrip(NoUnits, 2 * m * H.c * L)   # dimensionless coupling c'L

    # one Bogoliubov mode x = 2πj, in the stable form
    mode(x) = -2g^2 * N^2 / (sqrt(x^4 + 4g * N * x^2) + x^2 + 2g * N)

    # sum over j = ±1, …, ±jmax (the summand is even in j)
    S = 2 * sum(mode(2π * j) for j in 1:jmax)

    # tail |j| > jmax, using mode ≈ -g²N²/(2πj)² and Σ_{j>J} 1/j² ≈ 1/J - 1/(2J²) + 1/(6J³)
    S += -2 * g^2 * N^2 / (4π^2) * (1 / jmax - 1 / (2jmax^2) + 1 / (6jmax^3))

    return (g * N * (N - 1) + S) / (2 * m * L^2)
end

#=
Ground-state energy with N particles to first order in 1/c, from the mapping of the
strongly repulsive bosons to weakly attractive fermions (Bose-Fermi / Cheon-Shigehara
duality). Accurate for strong coupling γ = 2mc/n ≫ 1.

At c = ∞ the bosons are hard-core (Tonks-Girardeau) and the Bose-Fermi mapping
ψ_B = Π_{i<j} sgn(xᵢ - xⱼ) ψ_F maps them onto free fermions. At finite c the
fermions get a weak attractive interaction, and to first order in 1/c

    H_F = ∫₀ᴸ dx [ 1/(2m) ∂ₓψ†∂ₓψ - 1/(m²c) (∂ₓψ†)ψ†ψ(∂ₓψ) ]

with ψ now a fermion field. (In the textbook form -Σᵢ∂ᵢ² + 2c'Σ_{i<j}δ(xᵢ-xⱼ), c' = 2mc,
this is ∂ψ†∂ψ - (4/c') (∂ψ†)ψ†ψ(∂ψ).)

Unperturbed state: the Fermi sea with momenta kⱼ = 2π Iⱼ / L, Iⱼ = -(N-1)/2, …, (N-1)/2.
These are integers for N odd and half-integers for N even: the sgn factors make the
fermions antiperiodic on the ring when N is even. Then

    E_F = Σⱼ kⱼ² / (2m) = π² N(N²-1) / (6 m L²)

First-order correction, by Wick's theorem in the Fermi sea (⟨∂ψ†ψ⟩ = 0 by k → -k symmetry):

    ⟨(∂ψ†)ψ†ψ(∂ψ)⟩ = ⟨∂ψ†∂ψ⟩⟨ψ†ψ⟩ - |⟨∂ψ†ψ⟩|² = (Σⱼ kⱼ²/L)(N/L)

so ΔE = -L/(m²c) · N Σⱼ kⱼ² / L² = -(4N/g) E_F with g = 2mcL, i.e.

    E = E_F (1 - 4N/(2mcL)) + O(1/c²)

This agrees with expanding the Bethe equations of energy_exact to first order in 1/g,
and in the thermodynamic limit with e = (π²n³/6m)(1 - 4/γ).
Returns a Unitful energy in units of inverse length.
=#
function energy_bosonization(H::LiebLiniger, N::Integer)
    N ≥ 0  || throw(ArgumentError("N must be non-negative, got $N"))
    H.c > 0 || throw(ArgumentError("the 1/c expansion needs c > 0, got c = $(H.c)"))
    L, m = H.L, H.m
    g = ustrip(NoUnits, 2 * m * H.c * L)   # dimensionless coupling c'L

    # free fermions: Fermi sea with kⱼ L = 2π Iⱼ
    I = [j - (N + 1) / 2 for j in 1:N]
    E_F = sum(abs2, 2π .* I; init = 0.0) / (2 * m * L^2)   # = π² N(N²-1) / (6 m L²)

    # first-order correction from the attractive fermion interaction
    ΔE = -(4N / g) * E_F

    return E_F + ΔE
end

# ------------------------------------------------------------------------------
# Z2-graded CMPS type
# ------------------------------------------------------------------------------
#=
A uniform, single-species (one boson field) continuous MPS on a circle of
length L with periodic boundary conditions (PBC)

    |Ψ⟩ = Tr_aux[ P exp( ∫₀ᴸ dx  F/L ⊗ 𝟙 + R/sqrt(L) ⊗ ψ†(x) ) ] |Ω⟩

The trace over the auxiliary space closes the circle. F and R are dimensionless;
the length L of the circle is stored in the CMPS as a Unitful length (any unit).
L is a fixed parameter, not a variational degree of freedom.

F and R are 3-leg tensors on a Z2-graded virtual space V, with an extra
one-dimensional "physical" leg carrying the Z2 charge:

    F : V ← P0 ⊗ V     (P0 = Z2 charge 0, no particle created)
    R : V ← P1 ⊗ V     (P1 = Z2 charge 1, one particle created)
=#

const P0 = Z2Space(0 => 1)   # charge 0 leg
const P1 = Z2Space(1 => 1)   # charge 1 leg

struct CMPS{A<:AbstractTensorMap}
    F::A
    R::A
    L::Unitful.Length   # length of the circle, in any length unit (e.g. 10u"m", 2.5u"μm")
end

# Random CMPS on a Z2-graded virtual space V, e.g. rand_CMPS(Z2Space(0 => 2, 1 => 2), 10u"m")
function rand_CMPS(V::Z2Space, L; T = ComplexF64)
    F = randn(T, V ← P0 ⊗ V)
    R = randn(T, V ← P1 ⊗ V)
    return CMPS(F, R, L)
end

# Random CMPS with D0 even and D1 odd virtual states
rand_CMPS(D0::Integer, D1::Integer, L; T = ComplexF64) = rand_CMPS(Z2Space(0 => D0, 1 => D1), L; T = T)

# Basic accessors
virtualspace(ψ::CMPS) = codomain(ψ.F)[1]
bonddim(ψ::CMPS)      = dim(virtualspace(ψ))

# ------------------------------------------------------------------------------
# Quantum-state operations: |a⟩ + |b⟩, α|a⟩, ⟨a|b⟩, ‖a‖
# ------------------------------------------------------------------------------
# Two CMPS can only be combined if they live on the same circle
function check_length(a::CMPS, b::CMPS)
    a.L == b.L || throw(ArgumentError("CMPS lengths differ: $(a.L) vs $(b.L)"))
    return a.L
end

# Identity on V written as a 3-leg tensor e0 : V ← P0 ⊗ V, with the same leg
# structure as F. It plays the role of 𝟙 wherever F⊗𝟙 or F + c𝟙 appears.
e0(V) = isomorphism(ComplexF64, V ← P0 ⊗ V)

# Block-diagonal direct sum of two 3-leg tensors
#     ta : Va ← P ⊗ Va,   tb : Vb ← P ⊗ Vb   →   t : (Va⊕Vb) ← P ⊗ (Va⊕Vb)
# The one-dimensional charge leg P is shared, not summed.
function directsum(ta, tb)
    W = codomain(ta)[1] ⊕ codomain(tb)[1]
    P = domain(ta)[1]
    t = zeros(promote_type(scalartype(ta), scalartype(tb)), W ← P ⊗ W)
    for (c, b) in blocks(t)
        ba, bb = block(ta, c), block(tb, c)
        b[1:size(ba, 1), 1:size(ba, 2)] .= ba
        b[size(ba, 1)+1:end, size(ba, 2)+1:end] .= bb
    end
    return t
end

#=
|a⟩ + |b⟩: take the direct sum F = Fa ⊕ Fb, R = Ra ⊕ Rb. Block-diagonal matrices
stay block diagonal inside the path-ordered exponential, so the trace splits as
Tr_a[...] + Tr_b[...]. The bond dimension of the sum is Da + Db.
=#
Base.:+(a::CMPS, b::CMPS) = CMPS(directsum(a.F, b.F), directsum(a.R, b.R), check_length(a, b))

#=
α|a⟩: shift F → F + log(α) 𝟙. Since 𝟙 commutes with everything and
∫₀ᴸ dx log(α)/L = log(α), this multiplies the state by exp(log(α)) = α.
The bond dimension does not change. F (and R) become complex if they were real.
=#
function Base.:*(α::Number, a::CMPS)
    α == 0 && throw(ArgumentError("0 * CMPS is the zero vector, which is not a CMPS"))
    F = a.F + log(complex(α)) * e0(virtualspace(a))
    R = one(scalartype(F)) * a.R    # promote R to the element type of F
    return CMPS(F, R, a.L)
end
Base.:*(a::CMPS, α::Number) = α * a

Base.:-(a::CMPS) = (-1) * a
Base.:-(a::CMPS, b::CMPS) = a + (-b)

#=
⟨a|b⟩ = Tr(exp(L*T)) with the mixed transfer matrix (ket b, bra a)
    L*T = Fb⊗𝟙 + 𝟙⊗F̄a + Rb⊗R̄a
which is dimensionless because the state is built from F/L and R/sqrt(L).
L*T : Vb ⊗ Va' ← Vb ⊗ Va', so a and b may have different bond dimensions.
Like Julia's dot, the first argument is complex conjugated.
=#
function LinearAlgebra.dot(a::CMPS, b::CMPS)
    check_length(a, b)
    Ia, Ib = e0(virtualspace(a)), e0(virtualspace(b))
    @tensor LT[-1 -2; -3 -4] := b.F[-1; 1 -3] * conj(Ia[-2; 1 -4]) +
                                Ib[-1; 1 -3]  * conj(a.F[-2; 1 -4]) +
                                b.R[-1; 1 -3] * conj(a.R[-2; 1 -4])
    return tr(exp(LT))
end
# ‖a‖ = sqrt(⟨a|a⟩)
LinearAlgebra.norm(a::CMPS) = sqrt(real(dot(a, a)))


# ------------------------------------------------------------------------------
# expectation value of the energy
# ------------------------------------------------------------------------------
#=
E = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩ for the Lieb-Liniger Hamiltonian on the circle.

By translation invariance E = L × (energy density). With the dimensionless
transfer matrix L*T = F⊗𝟙 + 𝟙⊗F̄ + R⊗R̄ and the norm N = Tr(exp(L*T)),
the two densities are

    ⟨∂ψ†∂ψ⟩    = K / L³,   K = Tr(exp(L*T) [F,R]⊗conj([F,R])) / N
    ⟨ψ†ψ†ψψ⟩  = I / L²,   I = Tr(exp(L*T) RR⊗conj(RR)) / N

where K and I are dimensionless. The powers of L come from the F/L and
R/sqrt(L) in the definition of |Ψ⟩: [F/L, R/sqrt(L)] = [F,R]/L^(3/2) and
(R/sqrt(L))² = RR/L, each appearing once in the ket and once in the bra. Hence

    E = L [ K/(2m L³) + c I/L² ] = K/(2m L²) + c I/L

which is a Unitful quantity in units of inverse length (energy, with ħ = 1).
=#
function energy(ψ::CMPS, H::LiebLiniger)
    ψ.L == H.L || throw(ArgumentError("CMPS length $(ψ.L) differs from Hamiltonian length $(H.L)"))
    F, R, L = ψ.F, ψ.R, ψ.L

    # exp(L*T) and the norm ⟨ψ|ψ⟩ = Tr(exp(L*T))
    I0 = e0(virtualspace(ψ))
    @tensor LT[-1 -2; -3 -4] := F[-1; 1 -3]  * conj(I0[-2; 1 -4]) +
                                I0[-1; 1 -3] * conj(F[-2; 1 -4]) +
                                R[-1; 1 -3]  * conj(R[-2; 1 -4])
    expLT = exp(LT)
    N = tr(expLT)

    # kinetic term: commutator [F, R] : V ← P0 ⊗ P1 ⊗ V
    @tensor C[-1; -2 -3 -4] := F[-1; -2 1] * R[1; -3 -4] - R[-1; -3 1] * F[1; -2 -4]
    @tensor K = expLT[1 2; 3 4] * C[3; 5 6 1] * conj(C[4; 5 6 2])

    # interaction term: RR : V ← P1 ⊗ P1 ⊗ V
    @tensor RR[-1; -2 -3 -4] := R[-1; -2 1] * R[1; -3 -4]
    @tensor I = expLT[1 2; 3 4] * RR[3; 5 6 1] * conj(RR[4; 5 6 2])

    return real(K / N) / (2 * H.m * L^2) + H.c * real(I / N) / L
end

# ------------------------------------------------------------------------------
# expectation value of the particle number
# ------------------------------------------------------------------------------
#=
⟨N⟩ = ∫₀ᴸ dx ⟨ψ†(x)ψ(x)⟩ / ⟨ψ|ψ⟩ = L × (particle density), by translation invariance.

The density is ⟨ψ†ψ⟩ = Tr(exp(L*T) R⊗conj(R)) / (N L), where the 1/L comes from
the R/sqrt(L) in the definition of |Ψ⟩, appearing once in the ket and once in
the bra. Multiplying by L, the particle number

    ⟨N⟩ = Tr(exp(L*T) R⊗conj(R)) / N

is dimensionless and does not depend on L explicitly (a plain number).
=#
function particle_number(ψ::CMPS)
    F, R = ψ.F, ψ.R

    # exp(L*T) and the norm ⟨ψ|ψ⟩ = Tr(exp(L*T))
    I0 = e0(virtualspace(ψ))
    @tensor LT[-1 -2; -3 -4] := F[-1; 1 -3]  * conj(I0[-2; 1 -4]) +
                                I0[-1; 1 -3] * conj(F[-2; 1 -4]) +
                                R[-1; 1 -3]  * conj(R[-2; 1 -4])
    expLT = exp(LT)
    N = tr(expLT)

    # Tr(exp(L*T) R⊗conj(R))
    @tensor n = expLT[1 2; 3 4] * R[3; 5 1] * conj(R[4; 5 2])

    return real(n / N)
end

# ------------------------------------------------------------------------------
# Ground-state search with automatic differentiation
# ------------------------------------------------------------------------------
#=
OptimKit works in the space of variational parameters x = (F, R):
  * a step is x + α d, entry by entry on F and R,
  * the inner product is the Euclidean one on the entries of F and R.
These are NOT the quantum-state operations +, * and dot defined for CMPS above
(those would e.g. double the bond dimension at every step), so they are passed
to OptimKit explicitly below.

The cost function is the dimensionless effective energy (E - μ⟨N⟩)·L, and
Zygote computes its gradient with respect to F and R. For complex F and R, Zygote returns
∂/∂Re + i ∂/∂Im, which is the gradient for the inner product real(dot(⋅,⋅)).
=#

# Parameter-space operations on x = (F, R) and on gradients d = (dF, dR)
param_retract(x, d, α)  = ((x[1] + α * d[1], x[2] + α * d[2]), d)
param_inner(x, d1, d2)  = real(dot(d1[1], d2[1]) + dot(d1[2], d2[2]))
param_scale!(d, α)      = (α * d[1], α * d[2])
param_add!(d1, d2, α)   = (d1[1] + α * d2[1], d1[2] + α * d2[2])
param_transport!(d, x, η, α, xnew) = d     # flat parameter space: nothing to do

#=
Minimise the effective energy E - μ⟨N⟩ over F and R, starting from ψ0, where
E = ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ and ⟨N⟩ is the particle number. The chemical potential μ
(inverse length, e.g. 1u"m^-1") controls how many particles the optimum has.

Returns the optimised CMPS, its energy E = energy(ψ, H) WITHOUT the -μ⟨N⟩ term
(Unitful, inverse length) and the norm of the final gradient. Use
particle_number(ψ) for ⟨N⟩. Pass a different OptimKit algorithm with `alg`,
e.g. alg = LBFGS(30; maxiter = 2000, gradtol = 1e-10, verbosity = 3).

With verbosity = 3, OptimKit prints every iteration. Note that the printed
f is the dimensionless effective energy (E - μ⟨N⟩)·L, and ‖∇f‖ its gradient.
=#
function find_groundstate(ψ0::CMPS, H::LiebLiniger, μ::Unitful.Wavenumber;
                          alg = LBFGS(20; maxiter = 1000, gradtol = 1e-8, verbosity = 3))
    ψ0.L == H.L || throw(ArgumentError("CMPS length $(ψ0.L) differs from Hamiltonian length $(H.L)"))
    L = H.L

    # dimensionless effective energy (E - μ⟨N⟩)·L as a function of the parameters (F, R)
    function cost(x)
        ψ = CMPS(x[1], x[2], L)
        return ustrip(NoUnits, (energy(ψ, H) - μ * particle_number(ψ)) * L)
    end

    ψ, gnorm = minimize_cost(cost, ψ0, alg)
    return ψ, energy(ψ, H), gnorm
end

#=
Minimise E + λ (⟨N⟩ - N_desired)² over F and R, starting from ψ0: the energy
E = ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ plus a penalty that pushes the particle number ⟨N⟩ towards
N_desired (a plain number). λ (inverse length, an energy) sets how strongly.

The squared penalty is smooth, so the gradient goes to zero at the optimum and
LBFGS converges normally. The price is that ⟨N⟩ does not hit N_desired exactly:
at the optimum ⟨N⟩ - N_desired ≈ -μ/(2λ), with μ = dE/dN the chemical potential
at that particle number. Increase λ to get closer (at the cost of a stiffer,
slower optimisation).

Returns the optimised CMPS, its energy E = energy(ψ, H) WITHOUT the penalty
(Unitful, inverse length) and the norm of the final gradient. Use
particle_number(ψ) to see how close ⟨N⟩ ended up to N_desired.

With verbosity = 3, OptimKit prints every iteration. Note that the printed
f is the dimensionless (E + λ(⟨N⟩ - N_desired)²)·L, and ‖∇f‖ its gradient.
=#
function find_groundstate(ψ0::CMPS, H::LiebLiniger, N_desired::Real;
                          λ::Unitful.Wavenumber = 100u"m^-1",
                          alg = LBFGS(20; maxiter = 1000, gradtol = 1e-8, verbosity = 3))
    ψ0.L == H.L || throw(ArgumentError("CMPS length $(ψ0.L) differs from Hamiltonian length $(H.L)"))
    L = H.L

    # dimensionless penalised energy (E + λ(⟨N⟩ - N_desired)²)·L
    function cost(x)
        ψ = CMPS(x[1], x[2], L)
        return ustrip(NoUnits, (energy(ψ, H) + λ * (particle_number(ψ) - N_desired)^2) * L)
    end

    ψ, gnorm = minimize_cost(cost, ψ0, alg)
    return ψ, energy(ψ, H), gnorm
end

#=
Shared by both find_groundstate methods: minimise a cost function of the
parameters x = (F, R) with OptimKit, using Zygote for the gradient.
Returns the optimised CMPS and the norm of the final gradient.
=#
function minimize_cost(cost, ψ0::CMPS, alg)
    # value and gradient, in the form OptimKit expects
    function fg(x)
        f, grads = Zygote.withgradient(cost, x)
        g = grads[1]
        gF = g[1] === nothing ? zero(x[1]) : g[1]
        gR = g[2] === nothing ? zero(x[2]) : g[2]
        return f, (gF, gR)
    end

    x, f, g, numfg, history = optimize(fg, (ψ0.F, ψ0.R), alg;
                                       retract = param_retract, inner = param_inner,
                                       scale! = param_scale!, add! = param_add!,
                                       transport! = param_transport!)

    return CMPS(x[1], x[2], ψ0.L), sqrt(param_inner(x, g, g))
end
