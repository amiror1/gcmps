#=
gcmps_functions.jl — Gauged continuous Matrix Product States (cMPS) with U(1) symmetry

Variational ground-state search for a 1+1D U(1) gauge theory of two species of
charged bosons on a ring of length L:
    species a : charge +1
    species b : charge -1

cMPS parametrisation (in units of the lattice/UV scale a, so all quantities are
dimensionless and the ring length is Lovera = L/a):
    F    ~ the "Q" matrix of a standard cMPS (generator of evolution along x,
           carries trivial charge)
    Ja⁺  ~ creates/annihilates a particle, shifts the virtual U(1) charge by +1
    Jb⁺  ~ creates/annihilates b particle, shifts the virtual U(1) charge by -1

Every tensor has three legs  T[left_virtual, physical, right_virtual].
The "physical" leg is a 1-dimensional U(1) space whose only role is to carry
the charge of the operator (0 for F, ±1 for Ja⁺/Jb⁺), so TensorKit enforces
charge conservation block by block.

Gauge structure: the U(1) charge sitting on the virtual bond at position x is
interpreted as the electric field E(x). Because Ja⁺/Jb⁺ shift that charge by ±1,
Gauss's law (dE/dx = charge density) is built into the ansatz.


This file holds only the functions. The run scripts (G-CMPS.jl, density_vs_D.jl,
lieb_liniger_correlators.jl) and the tests in test/ load it with
include("gcmps_functions.jl"); set_virtual_space! sets the globals below.

Globals the functions read (set by the caller before use):
    Lovera            ring length L/a
    virtual_Q         charge (electric-field) operator on the virtual space
    virtual_identity  identity on the virtual space
=#

using TensorKit
using Zygote
using OptimKit
using LinearAlgebra: eigvals

"""
    compute_TM(F, Ja⁺, Jb⁺)

Build the cMPS transfer-matrix generator

    TM = F ⊗ 1̄ + 1 ⊗ F̄ + Ja⁺ ⊗ conj(Ja⁺) + Jb⁺ ⊗ conj(Jb⁺)

as a 4-leg tensor TM[ket_in, bra_in; ket_out, bra_out]. The transfer matrix
over a distance Δx (in units of a) is `exp(Δx * TM)`. Physical legs are
contracted between ket and bra (index 1).

`id` is the virtual identity; it defaults to the global `virtual_identity` and
only needs passing for a state on a different bond space (see `embed_state`).
"""
function compute_TM(F, Ja⁺, Jb⁺; id = virtual_identity)
    @tensor TM[-1,-2;-3,-4] := F[-1,1,-3]*conj(id[-2,1,-4]) + id[-1,1,-3]*conj(F[-2,1,-4]) + Ja⁺[-1,1,-3]*conj(Ja⁺[-2,1,-4]) + Jb⁺[-1,1,-3]*conj(Jb⁺[-2,1,-4])
    return TM
end

"""
    compute_norm(F, Ja⁺, Jb⁺)

Norm ⟨Ψ|Ψ⟩ of the cMPS on a ring of length `Lovera` (periodic boundary
conditions): `tr(exp(Lovera * TM))`.
"""
function compute_norm(F, Ja⁺, Jb⁺)
    TM = compute_TM(F, Ja⁺, Jb⁺)
    return tr(exp(Lovera*TM))
end

# largest real part of the eigenvalues of a (square) transfer matrix
leading_eigenvalue(TM) = maximum(cb -> maximum(real, eigvals(cb[2])), blocks(TM))

"""
    correlation_length(F, Ja⁺, Jb⁺)

Correlation length ξ/a of the charged (single-particle) correlators, from the
gap of the transfer-matrix generator between sectors of q = q_ket - q_bra:

    ξ/a = 1 / (λ₀ - λ₁)

λ₀ is the leading eigenvalue (real part) of the neutral block q = 0, which sets the
norm; λ₁ that of the charged blocks q = ±1, through which ⟨ψa†(Δx) ψa(0)⟩ and
⟨ψa† W ψb†⟩ propagate (±1 are complex conjugates, the max just picks either).
This is not the neutral (e.g. density–density) length, which would be the gap
inside the q = 0 block.
"""
function correlation_length(F, Ja⁺, Jb⁺)
    TM = compute_TM(F, Ja⁺, Jb⁺)
    lead(q) = maximum(real, eigvals(block(TM, U1Irrep(q))))
    return 1 / (lead(0) - max(lead(1), lead(-1)))
end

"""
    compute_shifted_TM(F, Ja⁺, Jb⁺)

The transfer-matrix generator shifted by its leading eigenvalue,
`TM - λ0 * 1`, so the largest eigenvalue of `exp(Lovera * TM)` is 1.

Why: nothing fixes the overall scale of the state (F → F + c·1 only
multiplies |Ψ⟩ by a number), so λ0 drifts during the optimisation. 
"""
function compute_shifted_TM(F, Ja⁺, Jb⁺)
    TM = compute_TM(F, Ja⁺, Jb⁺)
    λ0 = Zygote.ignore_derivatives(() -> leading_eigenvalue(TM))
    return TM - λ0*id(domain(TM))
end

"""
    compute_ring_density(F, Ja⁺, Jb⁺)

The normalised ring transfer matrix ρ = exp(Lovera*TM) / tr(exp(Lovera*TM)), so a
local expectation value is ⟨O⟩ = tr(O ρ) (see `ring_expectation`). Computing it
once and passing it on (`ρ` keyword of the observables) avoids repeating the
eigenvalues and the exponential for every term.
"""
function compute_ring_density(F, Ja⁺, Jb⁺)
    expLTM = exp(Lovera*compute_shifted_TM(F, Ja⁺, Jb⁺))   # shifted so it cannot overflow
    return expLTM/tr(expLTM)
end

# ⟨O⟩ = tr(O ρ), contracted directly (no product matrix is formed)
ring_expectation(O, ρ) = @tensor O[1,2,3,4]*ρ[3,4,1,2]

"""
    compute_a_squared_energy_density(F, Ja⁺, Jb⁺; regulator = 200, ga = 0., c = 0.)

Dimensionless energy density (energy density × a²) of the normalised state.
Translation invariance lets each local term be evaluated at a single point:
`tr(O * exp(Lovera*TM)) / norm`.

Terms:
1. Electric energy  (ga)² ⟨E²⟩, where E is the virtual charge (`virtual_Q`).
2. Kinetic (hopping) energy ⟨∂ψ†∂ψ⟩ for each species, which is
   the commutator term  [F, J] ⊗ conj([F, J]).
3. Lieb-Liniger repulsion  c ⟨ψ†ψ†ψψ⟩ within each species, J² ⊗ conj(J)².
4. Regularity penalty  `regulator * [Ja⁺, Jb⁺] ⊗ conj([Ja⁺, Jb⁺])`.
   For multi-species bosonic cMPS the kinetic energy is only finite
   if the R-matrices commute, so this term pushes [Ja⁺, Jb⁺] → 0.

Keywords:
- `ga`        : gauge coupling × a (strength of the electric term)
- `c`         : contact repulsion × a (within each species, not between a and b)
- `regulator` : weight of the commutator penalty
- `ρ`         : the normalised ring transfer matrix (`compute_ring_density`);
                pass it when it is already computed, to reuse it
"""
function compute_a_squared_energy_density(F, Ja⁺, Jb⁺; regulator = 200, ga = 0., ρ = compute_ring_density(F, Ja⁺, Jb⁺),c=0.)
    output = 0.
    #Electric energy : expectation value of \int_0^L dx E^2(x) / norm :
    # (evaluated at one point; E is the charge on the virtual bond, Q ⊗ conj(Q))
    @tensor Qsquared[-1,-2;-3,-4] := virtual_Q[-1,1,-3]*conj(virtual_Q[-2,1,-4])
    output += ga^2*ring_expectation(Qsquared, ρ)

    #Hopping term for positive charges
    # [F,J] ⊗ conj([F,J]), with C[left, phys_F, phys_J, right] = FJ - JF
    for J in [Ja⁺, Jb⁺]
        @tensor C[-1,-2,-3,-4] := F[-1,-2,1]*J[1,-3,-4] - J[-1,-3,1]*F[1,-2,-4]
        @tensor tmp[-1,-2;-3,-4] := C[-1,1,2,-3]*conj(C[-2,1,2,-4])

        output += ring_expectation(tmp, ρ)
    end

    #Lieb-Liniger repulsion term:  c * ψ† ψ† ψ ψ = c * J^2 ⊗ conj(J)^2 
    # ket J·J and bra J̄·J̄: the bonds between the two J's (3 ket, 4 bra) are summed,
    # the physical legs 1, 2 join ket to bra
    for J in [Ja⁺, Jb⁺]
        @tensor repulsion[-1,-2;-3,-4] := J[-1,1,3]*J[3,2,-3]*conj(J[-2,1,4])*conj(J[4,2,-4])
        output += c*ring_expectation(repulsion, ρ)
    end

    # Regularity penalty:  [Ja⁺, Jb⁺] ⊗ conj([Ja⁺, Jb⁺])
    @tensor C[-1,-2,-3,-4] := Ja⁺[-1,-2,1]*Jb⁺[1,-3,-4] - Jb⁺[-1,-3,1]*Ja⁺[1,-2,-4]
    @tensor tmp[-1,-2;-3,-4] := C[-1,1,2,-3]*conj(C[-2,1,2,-4])

    output += regulator*ring_expectation(tmp, ρ)

    return output
end

"""
    compute_particle_density(F, Ja⁺, Jb⁺)

Dimensionless total particle density (density × a):
`⟨ψa†ψa⟩ + ⟨ψb†ψb⟩`, i.e. the expectation of `J ⊗ conj(J)` summed over species.
`ρ` as in `compute_a_squared_energy_density`.
"""
function compute_particle_density(F, Ja⁺, Jb⁺; ρ = compute_ring_density(F, Ja⁺, Jb⁺))
    output = 0.
    for J in [Ja⁺, Jb⁺]
        @tensor tmp[-1,-2;-3,-4] := J[-1,1,-3]*conj(J[-2,1,-4])
        output += ring_expectation(tmp, ρ)
    end
    return output
end

"""
    compute_dimensionless_effective_energy_density(F, Ja⁺, Jb⁺; ga = 0., c = 0., strategy = :grand_canonical, μ = 0., λ = 10., Na = 1., Nb = Na)

Cost function for the optimisation, per unit length like the energy density.
`strategy` selects how the particle number is controlled (n = na + nb is the
particle density × a):

- `:grand_canonical` :  E_eff = energy density - μ * n
  `μ` is the chemical potential; the density is whatever minimises E_eff.
- `:number_variance` :  one term per species,
      E_eff = energy density + (λ/L) ⟨(N̂a - Na)²⟩ + (λ/L) ⟨(N̂b - Nb)²⟩,   N̂a = ∫dx na(x)
  each = (λ/L) [(⟨N̂⟩ - target)² + Var(N̂)], see `quadratic_number_penalty` and
  `compute_number_moments`. `Na`, `Nb` (default `Na`) are the target *numbers* of
  particles of each species on the ring. ⟨(N̂ - N)²⟩ is a total over the ring
  (Var(N̂) grows like L), so it is divided by L to be a penalty per unit length,
  in the units of the energy density: as a total energy, L·E_eff = E_total +
  λ⟨(N̂a - Na)²⟩ + λ⟨(N̂b - Nb)²⟩, so λ is independent of the ring length.
"""
function compute_dimensionless_effective_energy_density(F, Ja⁺, Jb⁺; ga = 0., c = 0., strategy = :grand_canonical, μ = 0., λ = 10., Na = 1., Nb = Na)
    if strategy == :number_variance
        E = compute_a_squared_energy_density(F, Ja⁺, Jb⁺; ga = ga, c = c)
        Na_mean, Na²_mean = compute_number_moments(F, Ja⁺, Jb⁺, Ja⁺)
        # The ansatz forces Na = Nb (the bond charge Na - Nb returns to itself around
        # the ring), so the moments of b equal those of a: with equal targets the b term
        # equals the a term exactly, and is not computed again (2.6× faster at D = 25)
        # the penalties are totals over the ring: /L makes them per unit length, like E
        Nb == Na && return E + 2*quadratic_number_penalty(Na_mean, Na²_mean, Na, λ)/Lovera
        Nb_mean, Nb²_mean = compute_number_moments(F, Ja⁺, Jb⁺, Jb⁺)
        return E + (quadratic_number_penalty(Na_mean, Na²_mean, Na, λ) +
                    quadratic_number_penalty(Nb_mean, Nb²_mean, Nb, λ))/Lovera
    end

    ρ = compute_ring_density(F, Ja⁺, Jb⁺)   # shared by energy and density
    E = compute_a_squared_energy_density(F, Ja⁺, Jb⁺; ga = ga, c = c, ρ = ρ)
    n = compute_particle_density(F, Ja⁺, Jb⁺; ρ = ρ)

    strategy == :grand_canonical && return E - μ*n
    error("unknown strategy $(strategy); use :grand_canonical or :number_variance")
end


"""
    dimensionless_two_point_function_a_a_times_L(F, Ja⁺, Jb⁺, Δxovera)

single-particle correlator of species a, ⟨ψa†(Δx) ψa(0)⟩, on the
ring. `Ja⁺` is inserted in the ket at one point and in the bra at a point a
distance `Δxovera` (= Δx/a) away.
"""
function dimensionless_two_point_function_a_a_times_L(F, Ja⁺, Jb⁺, Δxovera)
    TM = compute_shifted_TM(F, Ja⁺, Jb⁺)   # shifted so exp(L*TM) cannot overflow
    norm_state = tr(exp(Lovera*TM))

    return @tensor Ja⁺[1,4,3]*conj(Ja⁺[6,4,7])*exp(Δxovera*TM)[3,2,5,6]*exp((Lovera-Δxovera)*TM)[5,7,1,2]/norm_state
end

"""
    dimensionless_density_density_a(F, Ja⁺, Jb⁺, Δxovera)

Density-density correlator of species a, ⟨na(Δx) na(0)⟩ (times a²), on the ring:
`tr(N e^{Δx TM} N e^{(L-Δx) TM}) / tr(e^{L TM})` with N = Ja⁺ ⊗ conj(Ja⁺) the density
operator. For Δx > 0 this is ⟨ψa†(Δx) ψa†(0) ψa(0) ψa(Δx)⟩; as Δx → 0 it tends to
⟨ψa†ψa†ψaψa⟩ (`compute_a_pair_density`), without the contact term na δ(Δx).
"""
function dimensionless_density_density_a(F, Ja⁺, Jb⁺, Δxovera)
    TM = compute_shifted_TM(F, Ja⁺, Jb⁺)   # shifted so exp(L*TM) cannot overflow
    @tensor N[-1,-2;-3,-4] := Ja⁺[-1,1,-3]*conj(Ja⁺[-2,1,-4])
    return tr(N*exp(Δxovera*TM)*N*exp((Lovera-Δxovera)*TM))/tr(exp(Lovera*TM))
end

"""
    compute_a_pair_density(F, Ja⁺, Jb⁺, J)

Local pair density ⟨ψ†ψ†ψψ⟩ (times a²) of the species of `J` (`Ja⁺` or `Jb⁺`), the
expectation of J² ⊗ conj(J)², as in the repulsion term of
`compute_a_squared_energy_density`. Divided by n², it is g₂(0).
"""
function compute_a_pair_density(F, Ja⁺, Jb⁺, J; ρ = compute_ring_density(F, Ja⁺, Jb⁺))
    @tensor pair[-1,-2;-3,-4] := J[-1,1,3]*J[3,2,-3]*conj(J[-2,1,4])*conj(J[4,2,-4])
    return ring_expectation(pair, ρ)
end

# ℙ_q t: keep only the block of `t` (a map on the doubled ket ⊗ bra space) with
# coupled charge q = q_ket - q_bra, zero the others
function project_sector(t, q)
    P = zeros(ComplexF64, codomain(t), domain(t))
    for (c, b) in blocks(P)
        c == U1Irrep(q) && (block(P, c) .= one(b))
    end
    return P*t
end

"""
    dimensionless_a_W_b_function(F, Ja⁺, Jb⁺, Δxovera)

Gauge-invariant correlator ⟨ψa†(x) W_{x,y} ψb†(y)⟩ with y - x = Δxovera (in
units of a): an a particle at x and a b particle at y, joined by the Wilson line
W_{x,y} (a flux string). Reads

    tr[ℙ₀ e^{T(L-y)} (1 ⊗ J̄b) ℙ₋₁ e^{T(y-x)} (1 ⊗ J̄a) e^{Tx}] / tr[ℙ₀ e^{TL}]

where ℙ_q projects onto the sector with q_ket - q_bra = q. Both J̄ sit in the bra:
J̄a raises the bra charge by 1, so between x and y the state is in sector -1, and
J̄b brings it back to 0. The trace is cyclic, so e^{T(L-y)} and e^{Tx} merge into
e^{T(L-Δx)} and the result depends only on Δx.

Note the normalisation is the neutral sector only, tr[ℙ₀ e^{TL}], not the full
trace `compute_norm` uses.
"""
function dimensionless_a_W_b_function(F, Ja⁺, Jb⁺, Δxovera)
    TM = compute_shifted_TM(F, Ja⁺, Jb⁺)   # shifted so exp(L*TM) cannot overflow
    norm_state = tr(project_sector(exp(Lovera*TM), 0))

    E_string = project_sector(exp(Δxovera*TM), -1)            # x → y, inside the flux string
    E_rest   = project_sector(exp((Lovera-Δxovera)*TM), 0)    # y → x around the ring
    # joins the physical legs of J̄a (charge +1) and J̄b (charge -1); total charge 0
    w = ones(ComplexF64, U₁Space(1 => 1) ⊗ U₁Space(-1 => 1))

    return @tensor conj(Ja⁺[6,4,7])*E_string[5,7,1,2]*conj(Jb⁺[2,8,3])*E_rest[1,3,5,6]*w[4,8]/norm_state
end

#wrapped effective_energy function for optimkit
"""
    fg(F, Ja⁺, Jb⁺; kwargs...)

Return `(E_eff, ∇E_eff)` in the format OptimKit expects. The gradient is
taken with Zygote with respect to the tuple `(F, Ja⁺, Jb⁺)`; the real part of
the cost is used so the gradient is well defined for complex tensors.
All keywords (`ga`, `c`, `strategy`, `μ`, `λ`, `Na`, `Nb`) are passed on to
`compute_dimensionless_effective_energy_density`.
`withgradient` returns the value from the same forward pass as the gradient,
so the cost is not evaluated twice.
"""
function fg(F, Ja⁺, Jb⁺; kwargs...)
    E_eff, grads = Zygote.withgradient(xi->real(compute_dimensionless_effective_energy_density(xi...; kwargs...)), (F, Ja⁺, Jb⁺))
    grad_E_eff = grads[1]

    return E_eff, grad_E_eff
end

"""
    compute_number_moments(F, Ja⁺, Jb⁺, J)

First two moments `(⟨N̂⟩, ⟨N̂²⟩)` of the particle number of one species on the
ring, N̂ = ∫dx n(x), for the species of `J` (`Ja⁺` or `Jb⁺`); differentiable with
Zygote (used by `:number_variance`).

With N = J ⊗ conj(J) the transfer-matrix operator of the density n,
    ⟨N̂⟩  = L ⟨n⟩
    ⟨N̂²⟩ = ∫∫ dx dy ⟨n(x) n(y)⟩ = L ∫₀^L dΔ ⟨:n(Δ) n(0):⟩ + ⟨N̂⟩
where the last term is the contact term (n(x)n(y) = :n(x)n(y): + δ(x-y) n(x)).
The integral over Δ is exact with one matrix exponential (Van Loan's trick):

    exp(L [T N; 0 T]) = [e^{LT}  ∫₀^L e^{(L-s)T} N e^{sT} ds;  0  e^{LT}]

so tr(N · upper-right block) / tr(e^{LT}) = ∫₀^L dΔ ⟨:n(Δ) n(0):⟩. The 2×2 block
structure is a 2-dim auxiliary space; one exponential per species is cheaper than
one with a 3-dim space for both (16 vs 27 in units of one exponential). Numerical
quadrature of `dimensionless_density_density_a` (e.g. QuadGK) gives the same integral,
but is slower and has no Zygote rule.
"""
function compute_number_moments(F, Ja⁺, Jb⁺, J)
    TM = compute_shifted_TM(F, Ja⁺, Jb⁺)   # shifted so exp(L*TM) cannot overflow
    @tensor N[-1,-2;-3,-4] := J[-1,1,-3]*conj(J[-2,1,-4])
    A, E12, E21, E11 = Zygote.ignore_derivatives(() ->
        (U₁Space(0 => 2), _aux_unit(1, 2), _aux_unit(2, 1), _aux_unit(1, 1)))
    X = exp(Lovera*(TM ⊗ id(A) + N ⊗ E12))
    norm_state = real(tr(X*(id(domain(TM)) ⊗ E11)))     # tr e^{LT}
    N_mean = Lovera*real(tr(X*(N ⊗ E11)))/norm_state     # L ⟨n⟩ = L tr(N e^{LT})/tr
    pair   = real(tr(X*(N ⊗ E21)))/norm_state            # ∫₀^L dΔ ⟨:n(Δ)n(0):⟩
    
    return N_mean, Lovera*pair + N_mean
end

# |i⟩⟨j| on the 2-dim auxiliary space (trivial charge) of compute_number_moments
function _aux_unit(i, j)
    A = U₁Space(0 => 2)
    E = zeros(ComplexF64, A, A)
    block(E, U1Irrep(0))[i, j] = 1
    return E
end

"""
    quadratic_number_penalty(N̂_mean, N̂²_mean, N_target, λ)

λ ⟨(N̂ - N_target)²⟩ = λ (⟨N̂²⟩ - 2 N_target ⟨N̂⟩ + N_target²)
                    = λ [(⟨N̂⟩ - N_target)² + Var(N̂)],
from the moments of `compute_number_moments`. It pins the mean number and also
suppresses its fluctuations.
"""
quadratic_number_penalty(N̂_mean, N̂²_mean, N_target, λ) = λ*(N̂²_mean - 2*N_target*N̂_mean + N_target^2)


"""
    metric_preconditioner(; δmin = 1e-5, δmax = 1e-1)

Preconditioner for OptimKit: `optimize(fg, x, alg; precondition = metric_preconditioner())`.

The plain gradient measures a change of the tensors by its Euclidean size, but what
matters is how much the *state* changes. To leading order that is the metric

    G(dX, dX) = Σ dX[i,p,j] conj(dX[i',p,j']) L[i,i'] R[j,j']

with L, R the left/right environments of the bond: the ring density matrix
ρ = e^{L TM}/tr with its other ket-bra pair traced out (ρ ≈ R ⊗ L). Bond states with
tiny weight in L or R barely change the state, so their gradient is tiny and plain
L-BFGS crawls there. The natural gradient η = G⁻¹ g, i.e. η = L⁻¹ g R⁻¹ for each of
F, Ja⁺, Jb⁺, removes that disparity. L, R are normalised to unit trace and
regularised with δ·(largest eigenvalue) before inverting, since their smallest
eigenvalues are ~1e-14.

δ follows the norm of the vector being preconditioned, clamped to [δmin, δmax]
(as in Hauru, Van Damme, Haegeman, SciPost Phys. 10, 040 (2021))
"""
function metric_preconditioner(; δmin = 1e-5, δmax = 1e-1)
    cache = Ref{Any}(nothing)   # (x, L, R) at the last point
    return function (x, g)
        (cache[] === nothing || cache[][1] !== x) && (cache[] = (x, _metric_environments(x)...))
        _, L, R = cache[]
        δ = clamp(sqrt(real(sum(t -> dot(t, t), g))), δmin, δmax)
        Linv, Rinv = _regularised_inverse(L, δ), _regularised_inverse(R, δ)
        return map(t -> (@tensor η[-1,-2;-3] := Linv[-1,1]*t[1,-2,2]*Rinv[2,-3]), g)
    end
end

# left/right bond environments from the ring density matrix, Hermitian, unit trace
function _metric_environments(x)
    ρ = compute_ring_density(x...)
    @tensor R[-1; -2] := ρ[-1,-2,1,1]   # right bond (ket, bra), left pair traced out
    @tensor L[-2; -1] := ρ[1,1,-1,-2]   # left bond, transposed so that g = L η R
    normalise(t) = (t = (t + t')/2; t/real(tr(t)))
    return normalise(L), normalise(R)
end

_regularised_inverse(t, δ) = inv(t + δ*norm(t, Inf)*id(domain(t)))

#helpers for setting up bond spaces and starting points
###################
# identity on the bond space V, with the trivial physical leg attached
function _virtual_identity(V)
    id = zeros(V*U₁Space(0 => 1), V)
    for (c,b) in blocks(id)
        block(id, c) .= one(b)
    end
    return id
end

"""
    set_virtual_space!(V)

Set the globals the functions read for bond space `V`: the physical charge
spaces `Qa⁺`, `Qb⁺`, `Q_trivial`, the charge operator `virtual_Q` (each
charge-q block = q * identity) and `virtual_identity`.
"""
function set_virtual_space!(V)
    global Qa⁺ = U₁Space( 1 => 1)
    global Qb⁺ = U₁Space(-1 => 1)
    global Q_trivial = U₁Space( 0 => 1)
    global virtual_Q = zeros(V*Q_trivial, V)
    for (c,b) in blocks(virtual_Q)
        block(virtual_Q, c) .= c.charge*one(b)
    end
    global virtual_identity = _virtual_identity(V)
    return nothing
end

"""
    random_state(V)

Random `(F, Ja⁺, Jb⁺)` on bond space `V` (only symmetry-allowed blocks are filled).
"""
random_state(V) = (randn(ComplexF64, V*U₁Space(0 => 1), V),
                   randn(ComplexF64, V*U₁Space(1 => 1), V),
                   randn(ComplexF64, V*U₁Space(-1 => 1), V))

"""
    kick_J(x; scale = 0.1)

Add random noise of size `scale` to Ja⁺ and Jb⁺ of `x = (F, Ja⁺, Jb⁺)`.
J = 0 (the vacuum) is a stationary point of E_eff for every μ, since all
terms are quadratic in J, so a warm start from the vacuum never leaves it.
"""
kick_J(x; scale = 0.1) = (x[1],
                          x[2] + scale*randn(ComplexF64, codomain(x[2]), domain(x[2])),
                          x[3] + scale*randn(ComplexF64, codomain(x[3]), domain(x[3])))

"""
    embed_state(x, V_big; noise = 1e-3)

Embed a converged state `x = (F, Ja⁺, Jb⁺)` on a small bond space into the
larger bond space `V_big` (which must contain every charge sector of the small
one with at least the same degeneracy; an error otherwise). The old blocks go in
the top-left corner, everything else gets noise of size `noise`, so the new state
starts (almost) exactly at the old one and the optimiser can only improve on it.
"""
function embed_state(x, V_big; noise = 1e-3)
    V_small = domain(x[1])[1]
    for s in sectors(V_small)
        dim(V_big, s) >= dim(V_small, s) || error("embed_state: the new bond space must contain the old one, " *
            "but charge $(s.charge) has degeneracy $(dim(V_small, s)) in the old space and $(dim(V_big, s)) in the new one " *
            "(going to a smaller bond space is not possible)")
    end

    return map(x) do t
        W = codomain(t)[2]   # the 1-dim physical charge space of this tensor
        t_big = noise*randn(ComplexF64, V_big*W, V_big)
        for (c, b) in blocks(t)
            block(t_big, c)[1:size(b,1), 1:size(b,2)] .+= b
        end
        return t_big
    end
end

