using TensorKit
using OptimKit
using LinearAlgebra
using Zygote
using Unitful
using UnitfulChainRules

# ---
# Two-species Lieb Liniger Hamiltonian type
# ---
#=
Two-species Lieb-Liniger Hamiltonian on a circle of length L (PBC), in units ħ = 1
and with mass in inverse length. Two boson fields ψ_a and ψ_b with equal mass m and
an on-site repulsion c between particles of the same species only:

    H = ∫₀ᴸ dx [ 1/(2m) ∂ₓψ_a†∂ₓψ_a + 1/(2m) ∂ₓψ_b†∂ₓψ_b
                 + c ( ψ_a†ψ_a†ψ_aψ_a + ψ_b†ψ_b†ψ_bψ_b ) ]

There is no interaction between a and b, so H is the sum of two independent copies
of the single-species Lieb-Liniger Hamiltonian, one for each species.

Units, with ψ_a(x), ψ_b(x) of dimension length^(-1/2):
  * m : inverse length (e.g. 1u"m^-1")
  * L : length         (e.g. 10u"m")
  * c : dimensionless. All terms then have units of energy/length = length⁻²,
        and the usual dimensionless Lieb-Liniger parameter of each species is
        γ = 2mc/n, with n the particle density of that species.

All reference energies below assume equal particle numbers N_a = N_b = N, so they
are twice the single-species energy with N particles.

The chemical potential μ is not part of H: find_groundstate fixes the particle
number and returns μ as its Lagrange multiplier ν.
=#
struct LiebLiniger
    m::Unitful.Wavenumber   # mass, in units of inverse length
    c::Real                 # same-species coupling, dimensionless (equal for a and b)
    L::Unitful.Length       # system size (circumference of the circle)
end

#=
Exact ground-state energy with N_a = N_b = N particles from the Bethe ansatz
(repulsive, c ≥ 0). The species are independent, so E = 2 × E_LL(N), with E_LL(N)
the single-species ground-state energy of N particles, computed as follows.

In first quantisation H = Σᵢ -∂ᵢ²/(2m) + 2c Σ_{i<j} δ(xᵢ - xⱼ)  (the factor 2 because
ψ†ψ†ψψ counts every pair twice). Multiplying by 2m gives the textbook form
-Σᵢ ∂ᵢ² + 2c' Σ_{i<j} δ(xᵢ - xⱼ) with c' = 2mc, whose eigenstates have quasi-momenta kⱼ
solving the Bethe equations (logarithmic form)

    kⱼ L + Σₗ 2 arctan((kⱼ - kₗ)/c') = 2π Iⱼ,   Iⱼ = -(N-1)/2, ..., (N-1)/2  (ground state)

with energy E = Σⱼ kⱼ² / (2m). In terms of the dimensionless xⱼ = kⱼ L and g = c'L = 2mcL:

    xⱼ + Σₗ 2 arctan((xⱼ - xₗ)/g) = 2π Iⱼ,    E = Σⱼ xⱼ² / (2m L²)

These are solved with Newton's method, starting from the hard-core (g → ∞)
solution xⱼ = 2π Iⱼ. Returns a Unitful energy in units of inverse length
(the total for both species).
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
        maximum(abs, dx) < tol * max(1, maximum(abs, x)) && return 2 * sum(abs2, x) / (2 * m * L^2)   # factor 2: two species
    end
    error("Bethe equations did not converge in $maxiter Newton steps (N = $N, 2mcL = $g)")
end

#=
Ground-state energy with N_a = N_b = N particles in the Bogoliubov approximation
(c ≥ 0), accurate for weak coupling γ = 2mc/n ≪ 1. The species are independent, so
E = 2 × E_Bog(N), with E_Bog(N) the single-species result for N particles below.

In the textbook form -Σᵢ ∂ᵢ² + 2c' Σ_{i<j} δ(xᵢ - xⱼ) with c' = 2mc, density n = N/L
and momenta k = 2πj/L on the ring (see e.g. arXiv:1502.03291, arXiv:1609.08865):

    E_Bog = c' N(N-1)/L + Σ_{j≠0} ½ [ √(k⁴ + 4c'n k²) - k² - 2c'n ]

and our energy is E_Bog/(2m). In terms of x = kL = 2πj and g = c'L = 2mcL:

    E = [ g N(N-1) + Σ_{j≠0} ½ (√(x⁴ + 4gN x²) - x² - 2gN) ] / (2m L²)

The summand is evaluated in the equivalent, numerically stable form
-2g²N² / (√(x⁴ + 4gN x²) + x² + 2gN), which behaves as -g²N²/x² for large x.
The sum is done exactly up to |j| = jmax and the tail beyond is added analytically.
Returns a Unitful energy in units of inverse length (the total for both species).
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

    return 2 * (g * N * (N - 1) + S) / (2 * m * L^2)   # factor 2: two species
end

#=
Ground-state energy with N_a = N_b = N particles to first order in 1/c. The species
are independent, so E = 2 × E(N), with E(N) the single-species result for N
particles below, obtained from the mapping of the
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
Returns a Unitful energy in units of inverse length (the total for both species).
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

    return 2 * (E_F + ΔE)   # factor 2: two species
end

# ------------------------------------------------------------------------------
# U(1)-graded CMPS type
# ------------------------------------------------------------------------------
#=
A uniform, two-species (two boson fields ψ_a and ψ_b) continuous MPS on a circle of
length L with periodic boundary conditions (PBC). The auxiliary space is two copies
of the same U(1)-graded virtual space V, i.e. V ⊗ V, and

    |Ψ⟩ = Tr_{V⊗V}[ P exp( ∫₀ᴸ dx  F/L ⊗ 𝟙 + (R_a⊗𝟙_V)/sqrt(L) ⊗ ψ_a†(x)
                                           + (𝟙_V⊗R_b)/sqrt(L) ⊗ ψ_b†(x) ) ] |Ω⟩

R_a acts only on the first copy of V and R_b only on the second, so

    [R_a⊗𝟙_V, 𝟙_V⊗R_b] = 0

holds by construction: the regularity condition for two commuting boson species
(finite kinetic energy in the continuum) is built into the ansatz. F acts on the
full V ⊗ V and can entangle the two copies, so the two species are still
correlated. The trace over the auxiliary space closes the circle. F, R_a and R_b
are the three variational degrees of freedom and are dimensionless; the length L
of the circle is stored in the CMPS as a Unitful length (any unit). L is a fixed
parameter, not a variational degree of freedom.

The tensors are

    F   : V⊗V ← V⊗V      (no particle created)
    R_a : V ← Pp1 ⊗ V     (Pp1 = U(1) charge +1, one a particle created; acts on copy 1)
    R_b : V ← Pm1 ⊗ V     (Pm1 = U(1) charge -1, one b particle created; acts on copy 2)

R_a and R_b carry an extra one-dimensional "physical" leg with the U(1) charge of
the particle they create. F does not need one: a symmetric tensor V⊗V ← V⊗V
conserves the total U(1) charge automatically.

The U(1) charge counts N_a - N_b. V ⊗ V carries the fused (total) charge of the
two copies. Every R_a raises it by 1 (through copy 1), every R_b lowers it by 1
(through copy 2), and F conserves the total charge but may move charge between
the copies. The trace forces the auxiliary state to return to its starting
charge, so the state only contains components with N_a = N_b, which is the
sector we are interested in. (It is important that this is ONE U(1) on V ⊗ V, not
a separate U(1) per copy: then F would conserve each copy's charge and the trace
would kill every term containing an R_a.)

If D = dim(V), the auxiliary space has dimension D², while R_a and R_b are only
D × D matrices.

STORED exp(L*T) AND NORM

Besides F, R_a, R_b and L, a CMPS stores exp(L*T) (the exponential of its
dimensionless transfer matrix with itself, see transfer_matrix) and the norm
N = ⟨ψ|ψ⟩ = Tr(exp(L*T)). Every expectation value needs them and computing exp(L*T)
is by far the most expensive step (O(Dt¹²)), so it is done exactly once per state,
in the constructor. Rules that keep this consistent:

  * The constructor CMPS(F, R_a, R_b, L) is the only public way to build a CMPS, and
    it always computes exp(L*T) and N from the tensors. (The other inner
    constructor, with the TRUSTED token, takes them as given; it is only for
    operations that know them exactly without an exp, such as α * ψ.)
  * The struct is immutable, so changing a tensor means building a new CMPS.
  * NEVER mutate the tensors of a CMPS in place (ψ.F .+= ..., rmul!(ψ.R_a, 2), ...):
    exp(L*T) and N would then describe a different state, silently.

Automatic differentiation: building the CMPS INSIDE the function that Zygote
differentiates (as the cost functions of the optimisation do) makes Zygote follow the
gradient through the constructor, so expectation values that read ψ.expLT and ψ.N
have the correct gradient with respect to F, R_a and R_b. A CMPS built OUTSIDE the
differentiated function must not be used inside it: its exp(L*T) would be a
constant for Zygote, and the gradient would be silently wrong.
=#

const Pp1 = U1Space(1 => 1)    # charge +1 leg
const Pm1 = U1Space(-1 => 1)   # charge -1 leg

# token for the inner constructor that takes exp(L*T) and N as given (see above)
struct Trusted end
const TRUSTED = Trusted()

# F (4 legs), R_a, R_b (3 legs) and exp(L*T) (8 legs) are tensors of different type,
# hence three type parameters
struct CMPS{TF<:AbstractTensorMap, TR<:AbstractTensorMap, TE<:AbstractTensorMap}
    F::TF
    R_a::TR
    R_b::TR
    L::Unitful.Length   # length of the circle, in any length unit (e.g. 10u"m", 2.5u"μm")
    expLT::TE           # exp(L*T), with L*T = transfer_matrix(F, R_a, R_b)
    N::Float64          # ⟨ψ|ψ⟩ = Tr(exp(L*T)), real and positive

    # the public constructor: exp(L*T) and N are always computed from the tensors
    function CMPS(F::AbstractTensorMap, R_a::AbstractTensorMap, R_b::AbstractTensorMap,
                  L::Unitful.Length)
        expLT = exp(transfer_matrix(F, R_a, R_b))
        return new{typeof(F), typeof(R_a), typeof(expLT)}(F, R_a, R_b, L, expLT, real(tr(expLT)))
    end

    # internal: exp(L*T) and N known exactly by the caller (e.g. α * ψ)
    function CMPS(F::AbstractTensorMap, R_a::AbstractTensorMap, R_b::AbstractTensorMap,
                  L::Unitful.Length, expLT::AbstractTensorMap, N::Real, ::Trusted)
        return new{typeof(F), typeof(R_a), typeof(expLT)}(F, R_a, R_b, L, expLT, N)
    end
end

# Printing: the tensors (exp(L*T) alone is ~100 MB at Dt = 9) are not shown
function Base.show(io::IO, ψ::CMPS)
    print(io, "CMPS(V = ", virtualspace(ψ), ", L = ", ψ.L, ")")
end
function Base.show(io::IO, ::MIME"text/plain", ψ::CMPS)
    V = virtualspace(ψ)
    println(io, "CMPS on V ⊗ V, two boson species (a, b)")
    println(io, "  V (one copy) = ", V, "   (Dt = ", dim(V), ")")
    println(io, "  L            = ", ψ.L)
    print(io,   "  ⟨ψ|ψ⟩        = ", ψ.N)
end

# Random CMPS on a U(1)-graded virtual space V (one copy), e.g. rand_CMPS(U1Space(-1 => 2, 0 => 4, 1 => 2), 10u"m")
function rand_CMPS(V::U1Space, L; T = ComplexF64)
    F   = randn(T, V ⊗ V ← V ⊗ V)
    R_a = randn(T, V ← Pp1 ⊗ V)
    R_b = randn(T, V ← Pm1 ⊗ V)
    return CMPS(F, R_a, R_b, L)
end

# Random CMPS with D virtual states in each charge sector -qmax, ..., qmax (of one copy of V)
rand_CMPS(D::Integer, qmax::Integer, L; T = ComplexF64) =
    rand_CMPS(U1Space(q => D for q in -qmax:qmax), L; T = T)

# One copy V of the virtual space
virtualspace(ψ::CMPS) = codomain(ψ.R_a)[1]

# Identities, playing the role of 𝟙 wherever F⊗𝟙, R_a⊗𝟙_V, F + c𝟙, ... appears:
#   e0(V) : V⊗V ← V⊗V   (identity on the full auxiliary space, same legs as F)
#   e1(V) : V ← V       (identity on one copy, the 𝟙_V next to R_a and R_b)
e0(V) = isomorphism(ComplexF64, V ⊗ V ← V ⊗ V)
e1(V) = isomorphism(ComplexF64, V ← V)

# ------------------------------------------------------------------------------
# Quantum-state operations: α|ψ⟩, ⟨ψ|φ⟩, ‖ψ‖  (no |ψ⟩ + |φ⟩, see below)
# ------------------------------------------------------------------------------
# Two CMPS can only be combined if they live on the same circle, and a CMPS can only
# be used with a Hamiltonian on the same circle
function check_length(ψ::CMPS, φ::CMPS)
    ψ.L == φ.L || throw(ArgumentError("CMPS lengths differ: $(ψ.L) vs $(φ.L)"))
    return ψ.L
end
function check_length(ψ::CMPS, H::LiebLiniger)
    ψ.L == H.L || throw(ArgumentError("CMPS length $(ψ.L) differs from Hamiltonian length $(H.L)"))
    return ψ.L
end

#=
|ψ⟩ + |φ⟩ is NOT available for this ansatz. For a single auxiliary space it is the
direct sum of all tensors, on Vψ ⊕ Vφ. Here the sum would live on
(Vψ⊗Vψ) ⊕ (Vφ⊗Vφ), which is not of the form W ⊗ W. Embedding it in W ⊗ W with
W = Vψ ⊕ Vφ does not work either: W ⊗ W also contains the cross sectors Vψ⊗Vφ and
Vφ⊗Vψ, which are left invariant by R_a⊗𝟙, 𝟙⊗R_b and a block-diagonal F, and so add
extra (non-zero) terms Tr_{Vψ⊗Vφ}[...] + Tr_{Vφ⊗Vψ}[...] to the state. Hence + and
binary - throw an error.
=#
Base.:+(ψ::CMPS, φ::CMPS) = throw(ArgumentError("|ψ⟩ + |φ⟩ is not yet implemented"))
Base.:-(ψ::CMPS, φ::CMPS) = ψ + (-φ)

#=
α|ψ⟩: shift F → F + log(α) 𝟙 (𝟙 the identity on V ⊗ V). Since 𝟙 commutes with
everything and ∫₀ᴸ dx log(α)/L = log(α), this multiplies the state by exp(log(α)) = α.
The bond dimension does not change. F, R_a and R_b become complex if they were real.

No new exp is needed: L*T contains F⊗𝟙 + 𝟙⊗F̄, so it shifts by
(log α + conj(log α)) 𝟙 = log|α|² 𝟙, which commutes with everything, hence
exp(L*T) → |α|² exp(L*T) and N → |α|² N (as it should: ⟨αψ|αψ⟩ = |α|² ⟨ψ|ψ⟩).
=#
function Base.:*(α::Number, ψ::CMPS)
    α == 0 && throw(ArgumentError("0 * CMPS is the zero vector, which is not a CMPS"))
    F = ψ.F + log(complex(α)) * e0(virtualspace(ψ))
    R_a = one(scalartype(F)) * ψ.R_a    # promote R_a, R_b to the element type of F
    R_b = one(scalartype(F)) * ψ.R_b
    return CMPS(F, R_a, R_b, ψ.L, abs2(α) * ψ.expLT, abs2(α) * ψ.N, TRUSTED)
end
Base.:*(ψ::CMPS, α::Number) = α * ψ

Base.:-(ψ::CMPS) = (-1) * ψ

#=
The mixed transfer matrix (ket φ, bra ψ)

    L*T = Fφ⊗𝟙 + 𝟙⊗F̄ψ + (R_aφ⊗𝟙)⊗conj(R_aψ⊗𝟙) + (𝟙⊗R_bφ)⊗conj(𝟙⊗R_bψ)

which is dimensionless because the state is built from F/L, R_a/sqrt(L) and
R_b/sqrt(L). The charge legs are contracted between ket and bra, so a particles
only pair with a particles and b with b.
L*T : (Vφ⊗Vφ) ⊗ (Vψ⊗Vψ)' ← (Vφ⊗Vφ) ⊗ (Vψ⊗Vψ)', with legs
[-1 -2 -3 -4; -5 -6 -7 -8]: -1, -2 (out) and -5, -6 (in) are copies 1, 2 of the
ket, -3, -4 and -7, -8 copies 1, 2 of the bra. ψ and φ may have different bond
dimensions. Used for ⟨ψ|φ⟩ = Tr(exp(L*T)) (like Julia's dot, the first argument is
complex conjugated).

With φ = ψ this is the transfer matrix of the state with itself,

    L*T = F⊗𝟙 + 𝟙⊗F̄ + R̃_a⊗conj(R̃_a) + R̃_b⊗conj(R̃_b),   R̃_a = R_a⊗𝟙_V,  R̃_b = 𝟙_V⊗R_b,

whose exponential the CMPS constructor stores (ψ.expLT, with ψ.N = Tr exp(L*T)) for
all expectation values. exp(L*T) is computed exactly (densely): L*T is a
Dt⁴ × Dt⁴ matrix (block diagonal in the U(1) charge), so this costs O(Dt¹²).

The form on the raw tensors (ket tensors first, then bra tensors) is what the
constructor uses, since the CMPS does not exist yet at that point;
transfer_matrix(F, R_a, R_b) is the transfer matrix of these tensors with themselves.
=#
function transfer_matrix(Fφ, R_aφ, R_bφ, Fψ, R_aψ, R_bψ)
    Vψ, Vφ = codomain(R_aψ)[1], codomain(R_aφ)[1]
    Iψ, Iφ = e0(Vψ), e0(Vφ)       # identities on Vψ⊗Vψ and Vφ⊗Vφ
    iψ, iφ = e1(Vψ), e1(Vφ)       # identities on one copy
    @tensor LT[-1 -2 -3 -4; -5 -6 -7 -8] :=
        Fφ[-1 -2; -5 -6] * conj(Iψ[-3 -4; -7 -8]) +
        Iφ[-1 -2; -5 -6] * conj(Fψ[-3 -4; -7 -8]) +
        R_aφ[-1; 1 -5] * iφ[-2; -6] * conj(R_aψ[-3; 1 -7]) * conj(iψ[-4; -8]) +
        iφ[-1; -5] * R_bφ[-2; 1 -6] * conj(iψ[-3; -7]) * conj(R_bψ[-4; 1 -8])
    return LT
end
transfer_matrix(F, R_a, R_b) = transfer_matrix(F, R_a, R_b, F, R_a, R_b)

function transfer_matrix(ψ::CMPS, φ::CMPS)
    check_length(ψ, φ)
    return transfer_matrix(φ.F, φ.R_a, φ.R_b, ψ.F, ψ.R_a, ψ.R_b)
end

# ⟨ψ|φ⟩. For φ === ψ it is the stored norm, otherwise a new exp is needed.
LinearAlgebra.dot(ψ::CMPS, φ::CMPS) =
    ψ === φ ? complex(ψ.N) : tr(exp(transfer_matrix(ψ, φ)))
# ‖ψ‖ = sqrt(⟨ψ|ψ⟩), from the stored norm. The max(⋅, 0) guards against a tiny
# negative ⟨ψ|ψ⟩ from rounding errors when the state is (close to) zero.
LinearAlgebra.norm(ψ::CMPS) = sqrt(max(ψ.N, 0))

# ------------------------------------------------------------------------------
# expectation value of the energy
# ------------------------------------------------------------------------------
#=
E = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩ for the two-species Lieb-Liniger Hamiltonian on the circle.

By translation invariance E = L × (energy density). With the dimensionless
transfer matrix L*T (see transfer_matrix), the densities of species σ = a, b are

    ⟨∂ψ_σ†∂ψ_σ⟩      = K_σ / L³,   K_σ = Tr(exp(L*T) [F,R̃_σ]⊗conj([F,R̃_σ])) / N
    ⟨ψ_σ†ψ_σ†ψ_σψ_σ⟩ = I_σ / L²,   I_σ = Tr(exp(L*T) R̃_σR̃_σ⊗conj(R̃_σR̃_σ)) / N

with R̃_a = R_a⊗𝟙_V, R̃_b = 𝟙_V⊗R_b the operators on V ⊗ V and N = Tr(exp(L*T)).
K_σ and I_σ are dimensionless.

These are never built as tensors on V ⊗ V, which would mostly contract identities:
  * [F, R̃_σ] does involve both copies (F acts on V ⊗ V), so it is built as
    V⊗V ← P ⊗ V⊗V directly from F and R_σ.
  * R̃_σR̃_σ = (R_σR_σ)⊗𝟙_V only acts on one copy, so R_σR_σ is contracted with
    exp(L*T) directly and the other copy is traced over in ket and bra.

The powers of L come from the F/L and R_σ/sqrt(L) in the definition of |Ψ⟩:
[F/L, R̃_σ/sqrt(L)] = [F,R̃_σ]/L^(3/2) and (R̃_σ/sqrt(L))² = R̃_σR̃_σ/L, each
appearing once in the ket and once in the bra. Hence

    E = Σ_σ L [ K_σ/(2m L³) + c I_σ/L² ] = Σ_σ [ K_σ/(2m L²) + c I_σ/L ]

which is a Unitful quantity in units of inverse length (energy, with ħ = 1).
exp(L*T) and N are the ones stored in ψ, so this costs no exp.
=#
function energy(ψ::CMPS, H::LiebLiniger)
    L = check_length(ψ, H)
    F, R_a, R_b = ψ.F, ψ.R_a, ψ.R_b
    expLT, N = ψ.expLT, ψ.N       # exp(L*T) and the norm ⟨ψ|ψ⟩ = Tr(exp(L*T))

    # kinetic terms: commutators [F, R_a⊗𝟙] and [F, 𝟙⊗R_b] : V⊗V ← P(±1) ⊗ V⊗V
    @tensor C_a[-1 -2; -3 -4 -5] := F[-1 -2; 1 -5] * R_a[1; -3 -4] - R_a[-1; -3 1] * F[1 -2; -4 -5]
    @tensor C_b[-1 -2; -3 -4 -5] := F[-1 -2; -4 1] * R_b[1; -3 -5] - R_b[-2; -3 1] * F[-1 1; -4 -5]
    @tensor K_a = expLT[1 2 3 4; 5 6 7 8] * C_a[5 6; 9 1 2] * conj(C_a[7 8; 9 3 4])
    @tensor K_b = expLT[1 2 3 4; 5 6 7 8] * C_b[5 6; 9 1 2] * conj(C_b[7 8; 9 3 4])

    # interaction terms: R_σR_σ : V ← P(±1) ⊗ P(±1) ⊗ V on its own copy; the other
    # copy is traced over (legs 2 = 6, 4 = 8 for a; legs 1 = 5, 3 = 7 for b)
    @tensor RR_a[-1; -2 -3 -4] := R_a[-1; -2 1] * R_a[1; -3 -4]
    @tensor RR_b[-1; -2 -3 -4] := R_b[-1; -2 1] * R_b[1; -3 -4]
    @tensor I_a = expLT[1 2 3 4; 5 2 7 4] * RR_a[5; 9 10 1] * conj(RR_a[7; 9 10 3])
    @tensor I_b = expLT[1 2 3 4; 1 6 3 8] * RR_b[6; 9 10 2] * conj(RR_b[8; 9 10 4])

    K = real(K_a / N) + real(K_b / N)
    I = real(I_a / N) + real(I_b / N)
    return K / (2 * H.m * L^2) + H.c * I / L
end

# ------------------------------------------------------------------------------
# expectation value of the particle number
# ------------------------------------------------------------------------------
#=
⟨N_σ⟩ = ∫₀ᴸ dx ⟨ψ_σ†(x)ψ_σ(x)⟩ / ⟨ψ|ψ⟩ = L × (density of species σ), by translation
invariance.

The density is ⟨ψ_σ†ψ_σ⟩ = Tr(exp(L*T) R̃_σ⊗conj(R̃_σ)) / (N L), where the 1/L comes
from the R_σ/sqrt(L) in the definition of |Ψ⟩, appearing once in the ket and once
in the bra. Multiplying by L, the particle numbers

    ⟨N_a⟩ = Tr(exp(L*T) R̃_a⊗conj(R̃_a)) / N,    ⟨N_b⟩ = Tr(exp(L*T) R̃_b⊗conj(R̃_b)) / N

are dimensionless and do not depend on L explicitly (plain numbers). Returns the
tuple (⟨N_a⟩, ⟨N_b⟩). By the U(1) symmetry of the CMPS every component of the state
has N_a = N_b, so the two numbers agree (up to rounding errors).
R_σ only acts on its own copy of V, so the other copy is traced over in ket and bra.
exp(L*T) and N are the ones stored in ψ, so this costs no exp.
=#
function particle_number(ψ::CMPS)
    R_a, R_b = ψ.R_a, ψ.R_b
    expLT, N = ψ.expLT, ψ.N       # exp(L*T) and the norm ⟨ψ|ψ⟩ = Tr(exp(L*T))

    # Tr(exp(L*T) R̃_σ⊗conj(R̃_σ)): copy 2 traced over for a, copy 1 for b
    @tensor n_a = expLT[1 2 3 4; 5 2 7 4] * R_a[5; 9 1] * conj(R_a[7; 9 3])
    @tensor n_b = expLT[1 2 3 4; 1 6 3 8] * R_b[6; 9 2] * conj(R_b[8; 9 4])

    return real(n_a / N), real(n_b / N)
end

# ------------------------------------------------------------------------------
# Ground-state search with automatic differentiation
# ------------------------------------------------------------------------------
#=
OptimKit works in the space of variational parameters x = (F, R_a, R_b):
  * a step is x + α d, entry by entry on F, R_a and R_b,
  * the inner product is the Euclidean one on the entries of F, R_a and R_b.
These are NOT the quantum-state operations +, * and dot defined for CMPS above
(those would e.g. double the bond dimension at every step), so they are passed
to OptimKit explicitly below.

The cost functions are dimensionless (energies times L), and Zygote computes their
gradient with respect to F, R_a and R_b. For complex tensors, Zygote returns ∂/∂Re + i ∂/∂Im, which is the
gradient for the inner product real(dot(⋅,⋅)).

Because of the U(1) symmetry of the CMPS, every state has ⟨N_a⟩ = ⟨N_b⟩ = N, so the
particle number per species is all that needs to be controlled.
=#

# Parameter-space operations on x = (F, R_a, R_b) and on gradients d = (dF, dR_a, dR_b)
param_retract(x, d, α)  = (map((xi, di) -> xi + α * di, x, d), d)
param_inner(x, d1, d2)  = real(sum(map(dot, d1, d2)))
param_scale!(d, α)      = map(di -> α * di, d)
param_add!(d1, d2, α)   = map((d1i, d2i) -> d1i + α * d2i, d1, d2)
param_transport!(d, x, η, α, xnew) = d     # flat parameter space: nothing to do

# ------------------------------------------------------------------------------
# Metric preconditioner (one-point metric)
# ------------------------------------------------------------------------------
#=
WHY A PRECONDITIONER: THE GEOMETRY OF THE PARAMETER SPACE

The variational parameters X = (F, R_a, R_b) are coordinates Xⁱ on a manifold 𝓜
(one real coordinate per real and imaginary part of each tensor entry). Every point
gives a state |ψ(X)⟩, so 𝓜 is embedded in Hilbert space, and Hilbert space induces a
metric on it, exactly like the induced metric of a surface embedded in flat space:

    ds² = ⟨dψ|dψ⟩ = Gᵢⱼ(X) dXⁱ dXʲ,    Gᵢⱼ = Re⟨∂ᵢψ|∂ⱼψ⟩.

Distances measured with Gᵢⱼ are physical: they say how much the STATE changes. The
Euclidean inner product param_inner (the metric δᵢⱼ) only says how much the numbers
in the tensors change, which depends on how the state happens to be parametrised.

Zygote returns ∂ᵢf, the components of the one-form df: a COVECTOR, which measures the
rate of change of f along a direction (df(v) = ∂ᵢf vⁱ) but is not itself a
direction. Under a change of coordinates it transforms with the inverse Jacobian,
while displacements dXⁱ transform with the Jacobian. To turn it into a step (a
VECTOR) its index has to be raised with a metric. Steepest descent with respect to
the physical distance ds, i.e. minimising df(v) at fixed G(v, v), gives the
"natural gradient"

    ηⁱ = Gⁱʲ ∂ⱼf     (Gⁱʲ the inverse metric).

Plain gradient descent / plain LBFGS raises the index with δᵢⱼ instead. Directions
that barely change the state (virtual states with tiny weight) then get tiny gradient
components and the optimiser crawls along them; the eigenvalues of Gᵢⱼ below span
more than five orders of magnitude, which is exactly this problem.

HOW OPTIMKIT USES IT

We keep the Euclidean inner product, so the gradient returned by fg is Zygote's ∂ᵢf
unchanged (and optimtest still applies), and pass the inverse metric as a
preconditioner: precondition(x, g) takes a gradient-like vector g at the point x and
returns η = G(x)⁻¹ g, of the same shape (F, R_a, R_b). LBFGS uses it as its initial
guess for the inverse Hessian, so its steps are measured in the physical metric. It
must be (close to) linear in g and positive definite: inner(x, g, η) > 0.

THE ONE-POINT METRIC

Changing one tensor X ∈ {F, R_a, R_b} by dX changes the state by an insertion of dX̃
at every point of the ring, with dX̃ = dF, dR_a⊗𝟙_V or 𝟙_V⊗dR_b on V ⊗ V. Keeping
only the part where the ket and bra insertions sit at the same point gives the
one-point metric

    G(dX, dX) = Tr(ρ dX̃ ⊗ conj(dX̃)),    ρ = exp(L*T) / Tr(exp(L*T)),

the same contraction pattern as every expectation value in this file. Write ρ[a b; c d]
with a, c the ket legs and b, d the bra legs (each a pair of copies, a = (a1, a2)).

For F (no charge leg):

    G(B, A) = Σ ρ[a b; c d] A[c; a] conj(B[d; b]),
    (G A)[d; b] = Σ_{a,c} ρ[a b; c d] A[c; a],

so G is ρ with its legs regrouped: M_F = permute(ρ, ((7,8,3,4), (5,6,1,2))) maps the
legs (c, a) of A to the legs (d, b) of G A.

For R_a, Ã = A⊗𝟙_V does not act on copy 2, so copy 2 is traced over in ket and bra:
ρ_a[a1 b1; c1 d1] = Σ_{x,y} ρ[a1 x b1 y; c1 x d1 y]. With this reduced ρ_a the metric
has exactly the single-copy form

    (G A)[d1; p b1] = Σ ρ_a[a1 b1; c1 d1] A[c1; p a1],   M_a = permute(ρ_a, ((4,2),(3,1))),

and the same for R_b with copy 1 traced over. With A regrouped as a vector (its
one-dimensional charge leg p kept as the domain), the natural gradient for each tensor
solves M η = g. TensorKit keeps each M block diagonal in the charge, so these are a
few dense linear solves (the largest, for F, of size D⁴; cheap next to exp(L*T)).

What is left out: the true tangent vector inserts dX everywhere on the ring, so
⟨dψ|dψ⟩ is a double integral over the two insertion points. The one-point metric is
its same-point part; the terms with different points (connected by exp((y-x)T)) are
dropped. A preconditioner only needs to be roughly right, the line search corrects
the step length.

DEGENERATE DIRECTIONS: THE ANALOGUE OF GAUGE FREEDOM

Some directions do not change the physical state: gauge transformations
F → gFg⁻¹, R_σ → gR_σg⁻¹ (with g = g_V⊗g_V acting on both copies), and virtual states
that the state does not use. Along them ds² ≈ 0, so G has (near-)zero eigenvalues and
Gⁱʲ does not exist; 𝓜 is fibred into gauge orbits, like configurations related by
diffeomorphisms in GR. Instead of fixing a gauge we regularise: invert M + δ ‖M‖ 𝟙,
i.e. add a small flat piece δ·δᵢⱼ to the metric, which gives the gauge directions a
small but finite step. As in Hauru, Van Damme, Haegeman, SciPost Phys. 10, 040 (2021),
δ follows the size of the vector being preconditioned, clamped to [δmin, δmax]:
strongly regularised far from the optimum, close to the true metric near convergence.
(This makes the map mildly nonlinear in g, which works fine in practice.)

The metric depends on the point (like g_μν(x)) through ρ, so it takes the CMPS ψ at
that point and uses its stored exp(L*T) and N, like energy and particle_number.
minimize_cost passes the CMPS that the cost function built at that point, so
exp(L*T) is not recomputed here.
=#
function metric_preconditioner(ψ::CMPS, g; δmin = 1e-5, δmax = 1e-1)
    hermitian(M) = (M + M') / 2             # remove rounding asymmetry: M is Hermitian
    x = (ψ.F, ψ.R_a, ψ.R_b)

    # one-point metrics at x, from ρ = exp(L*T)/Tr
    ρ = ψ.expLT / ψ.N
    @tensor ρ_a[-1 -2; -3 -4] := ρ[-1 1 -2 2; -3 1 -4 2]    # trace over copy 2
    @tensor ρ_b[-1 -2; -3 -4] := ρ[1 -1 2 -2; 1 -3 2 -4]    # trace over copy 1
    M_F = permute(ρ, ((7, 8, 3, 4), (5, 6, 1, 2)))
    M_a = permute(ρ_a, ((4, 2), (3, 1)))
    M_b = permute(ρ_b, ((4, 2), (3, 1)))
    Ms = map(hermitian, (M_F, M_a, M_b))

    # regularisation δ, following the size of g (Hauru, Van Damme, Haegeman 2021)
    δ = clamp(sqrt(param_inner(x, g, g)), δmin, δmax)
    Minvs = map(M -> inv(M + δ * norm(M, Inf) * id(domain(M))), Ms)

    # η = (M + δ‖M‖)⁻¹ g for each of F, R_a, R_b. F: regroup g[c1 c2; a1 a2] as a
    # vector on V⊗V⊗V'⊗V'. R: regroup g[c; p a] as a vector on V ⊗ V' with the
    # charge leg p as domain. Solve, and regroup back.
    gF, ga, gb = g
    ηF = permute(Minvs[1] * permute(gF, ((1, 2, 3, 4), ())), ((1, 2), (3, 4)))
    ηR(Minv, gi) = permute(Minv * permute(gi, ((1, 3), (2,))), ((1,), (3, 2)))
    return (ηF, ηR(Minvs[2], ga), ηR(Minvs[3], gb))
end

#=
Fixed-particle-number ground-state search with the augmented Lagrangian method.
N_desired is a plain number, the number of particles PER SPECIES (as in
energy_exact(H, N)).

Stage k minimises, over F, R_a and R_b, the dimensionless cost

    C_k = (E - ν_k Σ_σ (⟨N_σ⟩ - N_desired) + λ Σ_σ (⟨N_σ⟩ - N_desired)²)·L

with E = ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩, a fixed penalty weight λ and a Lagrange multiplier ν_k (both
inverse length, i.e. energies). By the U(1) symmetry ⟨N_a⟩ = ⟨N_b⟩ = N, so a single
multiplier, the same for both species, is enough.

Why: with the penalty alone (ν = 0) the minimum sits at N - N_desired ≈ -μ/(2λ),
with μ = dE/dN the chemical potential per species, so N = N_desired exactly would
need λ → ∞, which makes the cost stiff and the line search fail. At the minimum of
stage k

    N_k - N_desired = (ν_k - μ) / (2λ),

so after each stage the multiplier is updated as

    ν_{k+1} = ν_k - 2λ (N_k - N_desired).

Its fixed point is ν = μ, where N = N_desired exactly at FINITE λ. Each update
shrinks the error in N by roughly a factor κ/(κ + 2λ), with κ = dμ/dN, so a moderate
λ converges in a handful of stages while the cost stays well conditioned. ν
converges to μ = dE/d⟨N_σ⟩ ALONG THE VARIATIONAL FAMILY. A CMPS is not a particle-number
eigenstate (it has number fluctuations on the ring), so for small N this is not the
same as a finite difference of exact energies at fixed N, E(N+1) - E(N); it should
approach the thermodynamic chemical potential for large N.

ν cannot simply be added to the variational parameters: C is linear in ν and the
solution is a saddle point (minimum in ψ, MAXIMUM in ν). The update above is the
gradient ascent step in ν, done outside OptimKit, which only minimises.

Each stage is a separate LBFGS run at FIXED ν_k, started from the result of the
previous stage: LBFGS builds its curvature information from successive gradients,
which only makes sense if the cost function stays the same. Stages use the cheap,
loose alg_stage until |⟨N⟩ - N_desired| < Ntol; then one final run with alg, at the
converged ν, polishes the state. If Ntol is not reached within maxstages stages, a
warning is printed and the final run is done anyway.

One line is printed per stage (ν, energy, ⟨N_a⟩, gradient norm). Returns the
optimised CMPS, its energy E = energy(ψ, H) (Unitful, inverse length, without the
penalty and multiplier terms), the final multiplier ν (inverse length) and the
norm of the final gradient.

precondition = true (default) uses the metric preconditioner (see
metric_preconditioner); set it to false for plain LBFGS.
=#
function find_groundstate(ψ0::CMPS, H::LiebLiniger, N_desired::Real;
                          λ::Unitful.Wavenumber = 100u"m^-1",
                          ν0::Unitful.Wavenumber = 0.0u"m^-1",
                          Ntol::Real = 1e-5,
                          maxstages::Integer = 20,
                          precondition::Bool = true,
                          alg_stage = LBFGS(20; maxiter = 500, gradtol = 1e-4, verbosity = 1),
                          alg = LBFGS(20; maxiter = 1000, gradtol = 1e-8, verbosity = 1))
    L = check_length(ψ0, H)

    # dimensionless cost C as a function of x = (F, R_a, R_b), at fixed multiplier ν.
    # The CMPS is built here, inside the function Zygote differentiates, so its
    # exp(L*T) (computed once, shared by all terms) is differentiated too.
    # Returns (C, ψ): ψ is passed on to the preconditioner (see minimize_cost).
    function cost(x, ν)
        ψ = CMPS(x..., L)
        N_a, N_b = particle_number(ψ)
        ΔN_a, ΔN_b = N_a - N_desired, N_b - N_desired
        C = ustrip(NoUnits, (energy(ψ, H) - ν * (ΔN_a + ΔN_b) + λ * (ΔN_a^2 + ΔN_b^2)) * L)
        return C, ψ
    end

    # one stage: minimise C at fixed ν, report, and return the state and ⟨N⟩ - N_desired
    function stage(ψ, ν, alg, label)
        ψ, gnorm = minimize_cost(x -> cost(x, ν), ψ, alg; precondition)
        N_a, N_b = particle_number(ψ)
        println("  $label:  ν = ", round(typeof(ν), ν; sigdigits = 6), "   E = ", energy(ψ, H),
                "   ⟨N_a⟩ = ", round(N_a; digits = 6), "   ‖∇‖ = ", round(gnorm; sigdigits = 3))
        return ψ, (N_a + N_b) / 2 - N_desired, gnorm
    end

    ψ, ν = ψ0, float(ν0)
    converged = false
    for k in 1:maxstages
        ψ, ΔN, _ = stage(ψ, ν, alg_stage, "stage $k")
        ν -= 2λ * ΔN                          # multiplier update (gradient ascent in ν)
        converged = abs(ΔN) < Ntol
        converged && break
    end
    converged || @warn "find_groundstate: |⟨N⟩ - N_desired| < $Ntol not reached in $maxstages stages"

    ψ, _, gnorm = stage(ψ, ν, alg, "final")
    return ψ, energy(ψ, H), ν, gnorm
end

#=
Minimise a cost function of the parameters x = (F, R_a, R_b) with OptimKit, using
Zygote for the gradient and, if precondition = true, the metric preconditioner.
Returns the optimised CMPS and the norm of the final gradient (the Euclidean norm
of Zygote's gradient, also when preconditioning).

cost(x) must return the pair (C, ψ), with C the value to minimise and
ψ = CMPS(x..., L) the state it built (with its exp(L*T)). Zygote only differentiates
C; ψ is kept from the last evaluation and passed to the preconditioner, which needs
exp(L*T) at the same point. OptimKit calls precondition(x, g) with the very x that
fg was last evaluated at (the tensors are compared with ===), so ψ is normally
reused; otherwise (e.g. if the line search accepted an earlier point) a new CMPS is
built. The returned CMPS is obtained the same way.

The points x that OptimKit handles stay tuples of tensors, not CMPS: a CMPS built
outside fg (e.g. in param_retract) would compute an exp(L*T) that Zygote cannot
differentiate, and fg would have to compute it again anyway.
=#
function minimize_cost(cost, ψ0::CMPS, alg; precondition::Bool = true)
    L = ψ0.L
    last = Ref{Any}(ψ0)           # CMPS of the last cost evaluation

    # the CMPS at the point x: the last one evaluated if it was at x, else a new one
    function state(x)
        ψ = last[]
        ψ.F === x[1] && ψ.R_a === x[2] && ψ.R_b === x[3] && return ψ
        return last[] = CMPS(x..., L)
    end

    # value and gradient, in the form OptimKit expects
    function fg(x)
        (f, ψ), grads = Zygote.withgradient(cost, x)    # gradient of C only
        last[] = ψ
        g = map((gi, xi) -> gi === nothing ? zero(xi) : gi, grads[1], x)   # `nothing` → zero tensor
        return f, g
    end

    x, _, g, _, _ = optimize(fg, (ψ0.F, ψ0.R_a, ψ0.R_b), alg;
                             retract = param_retract, inner = param_inner,
                             scale! = param_scale!, add! = param_add!,
                             transport! = param_transport!,
                             precondition = precondition ? (x, g) -> metric_preconditioner(state(x), g) :
                                                           (x, g) -> deepcopy(g))

    return state(x), sqrt(param_inner(x, g, g))
end
