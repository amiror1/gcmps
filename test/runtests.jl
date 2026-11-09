#=
Tests for gcmps_functions.jl

Run from the project folder, in the Julia REPL:
    include("test/runtests.jl")

The first run compiles TensorKit/Zygote/OptimKit, so it takes a while; after
that the whole file should finish in well under a minute. OptimKit may print
"Linesearch not converged" warnings near convergence; these are harmless.
=#

using Test
using Random

include(joinpath(@__DIR__, "..", "gcmps_functions.jl"))

# (not `const`, so this also works in a REPL where G-CMPS.jl already defined them)
Qa⁺       = U₁Space( 1 => 1)
Qb⁺       = U₁Space(-1 => 1)
Q_trivial = U₁Space( 0 => 1)

# Set the globals the functions read (same construction as in G-CMPS.jl)
function setup_globals!(virtual_space, L)
    global virtual_Q = zeros(virtual_space*Q_trivial, virtual_space)
    for (c,b) in blocks(virtual_Q)
        block(virtual_Q, c) .= c.charge*one(b)
    end
    global virtual_identity = zeros(virtual_space*Q_trivial, virtual_space)
    for (c,b) in blocks(virtual_identity)
        block(virtual_identity, c) .= one(b)
    end
    global Lovera = L
    return nothing
end

random_tensors(virtual_space; scale = 1.) = (
    scale*randn(ComplexF64, virtual_space*Q_trivial, virtual_space),
    scale*randn(ComplexF64, virtual_space*Qa⁺,       virtual_space),
    scale*randn(ComplexF64, virtual_space*Qb⁺,       virtual_space),
)

# ⟨[A,B] ⊗ conj([A,B])⟩ written out as the original four-term expansion,
# to compare against the commutator form used in gcmps_functions.jl
function commutator_expanded(A, B, M)
    @tensor t[-1,-2;-3,-4] := A[-1,1,2]*conj(A[-2,1,3])*B[2,4,-3]*conj(B[3,4,-4])
    @tensor t[-1,-2;-3,-4] += A[2,4,-3]*conj(A[3,4,-4])*B[-1,1,2]*conj(B[-2,1,3])
    @tensor t[-1,-2;-3,-4] -= A[-1,1,2]*conj(A[3,1,-4])*B[2,4,-3]*conj(B[-2,4,3])
    @tensor t[-1,-2;-3,-4] -= A[2,4,-3]*conj(A[-2,4,3])*B[-1,1,2]*conj(B[3,1,-4])
    return tr(t*M)
end

function commutator_form(A, B, M)
    @tensor C[-1,-2,-3,-4] := A[-1,-2,1]*B[1,-3,-4] - B[-1,-3,1]*A[1,-2,-4]
    @tensor t[-1,-2;-3,-4] := C[-1,1,2,-3]*conj(C[-2,1,2,-4])
    return tr(t*M)
end

# species-a density alone (compute_a_particle_density returns na + nb)
function density_a(F, Ja⁺, Jb⁺)
    TM = compute_TM(F, Ja⁺, Jb⁺)
    @tensor tmp[-1,-2;-3,-4] := Ja⁺[-1,1,-3]*conj(Ja⁺[-2,1,-4])
    return tr(tmp*exp_neutral(Lovera, TM))/compute_norm(F, Ja⁺, Jb⁺)
end


@testset "gcmps_functions" begin

    V6 = U₁Space(-1 => 2, 0 => 2, 1 => 2)

    @testset "number of parameters per tensor" begin
        V16 = U₁Space(-3=>1, -2=>2, -1 => 3, 0 => 4, 1=>3 , 2=>2, 3=>1)
        @test dim.(random_tensors(V6))  == (12, 8, 8)
        @test dim.(random_tensors(V16)) == (44, 40, 40)
    end

    @testset "commutator form = four-term expansion" begin
        Random.seed!(1)
        F, Ja⁺, Jb⁺ = random_tensors(V6)
        M = randn(ComplexF64, V6⊗V6', V6⊗V6')   # stand-in for exp(L*TM)
        for (A, B) in [(F, Ja⁺), (F, Jb⁺), (Ja⁺, Jb⁺)]
            @test commutator_form(A, B, M) ≈ commutator_expanded(A, B, M) rtol = 1e-10
        end
        # same-species commutators vanish, so the regulator only needs the a–b pair
        @test abs(commutator_expanded(Ja⁺, Ja⁺, M)) < 1e-10
        @test abs(commutator_expanded(Jb⁺, Jb⁺, M)) < 1e-10
    end

    @testset "observables of a random state" begin
        Random.seed!(2)
        setup_globals!(V6, 10.)
        x = random_tensors(V6; scale = 0.3)

        norm_state = compute_norm(x...)
        E = compute_a_squared_energy_density(x...; ga = 1.)
        n = compute_particle_density(x...)

        # expectation values of Hermitian operators: real up to rounding
        @test abs(imag(norm_state)) < 1e-10*abs(norm_state)
        @test abs(imag(E)) < 1e-10*abs(E)
        @test abs(imag(n)) < 1e-10*abs(n)
        @test real(norm_state) > 0
        @test real(n) > 0

        # Gauss's law on a ring forces zero total charge, so na = nb
        @test density_a(x...) ≈ real(n)/2 rtol = 1e-8

        # correlator at zero separation is the species-a density
        @test dimensionless_two_point_function_a_a_times_L(x..., 0.) ≈ density_a(x...) rtol = 1e-8
    end

    #=
    TM keeps q = q_ket - q_bra, so it commutes with every sector projector ℙ_q.
    exp_neutral and the correlators rely on this: projecting after the exponential,
    exponentiating the projected TM, or exponentiating only the block must agree.
    =#
    @testset "sector projectors commute with the transfer matrix" begin
        Random.seed!(4)
        setup_globals!(V6, 10.)
        TM = compute_TM(random_tensors(V6; scale = 0.3)...)
        one_TM = id(domain(TM))
        ℙ(q) = project_sector(one_TM, q)
        qs = sort([c.charge for c in blocksectors(TM)])
        Δx = 2.

        @test qs == collect(-2:2)                     # V6 has charges -1..1
        @test sum(ℙ(q) for q in qs) ≈ one_TM
        for q in qs
            @test ℙ(q)*ℙ(q) ≈ ℙ(q)
            @test ℙ(q)*TM ≈ TM*ℙ(q)
            for q′ in qs
                q′ == q || @test norm(ℙ(q)*TM*ℙ(q′)) < 1e-12
            end
            # ℙ e^T is the same wherever ℙ goes, as long as one ℙ is outside the exponential
            P, T = ℙ(q), Δx*TM
            E = P*exp(T)
            @test E ≈ exp(T)*P
            @test E ≈ P*exp(T)*P
            @test E ≈ P*exp(P*T)
            @test E ≈ exp(P*T)*P
            @test E ≈ P*exp(T*P)
            @test E ≈ P*exp(P*T*P)*P
            @test E ≈ project_sector(exp(Δx*project_sector(TM, q)), q)   # as in exp_neutral
            @test E ≈ _sector_map(TM, q, exp(Δx*block(TM, U1Irrep(q))))
            # without the outer ℙ the other sectors give e^0 = 1, not 0
            @test exp(P*T) ≈ E + (one_TM - P)
        end
        @test exp_neutral(Lovera, TM) ≈ project_sector(exp(Lovera*TM), 0)
    end

    @testset "gradient matches finite difference ($(kw.strategy))" for kw in [
            (strategy = :grand_canonical, ga = 2.5, μ = 5.),
            (strategy = :number_variance, ga = 2.5, Na = 5., λ = 10.)]
        Random.seed!(3)
        setup_globals!(V6, 10.)
        x = random_tensors(V6; scale = 0.3)
        d = random_tensors(V6)        # random direction
        h = 1e-6

        _, g = fg(x...; kw...)
        E₊ = fg((x .+ h .* d)...; kw...)[1]
        E₋ = fg((x .- h .* d)...; kw...)[1]
        finite_diff = (E₊ - E₋)/(2h)
        autodiff    = sum(real(dot(gi, di)) for (gi, di) in zip(g, d))

        @test autodiff ≈ finite_diff rtol = 1e-5
    end

    @testset "unknown strategy raises an error" begin
        setup_globals!(V6, 10.)
        x = random_tensors(V6; scale = 0.3)
        @test_throws ErrorException compute_dimensionless_effective_energy_density(x...; strategy = :typo)
    end

    # the regulator keyword reaches the energy: E_eff changes by regulator × ⟨[Ja⁺,Jb⁺]†[Ja⁺,Jb⁺]⟩
    @testset "regulator weight reaches the cost ($(strategy))" for strategy in [:grand_canonical, :number_variance]
        Random.seed!(6)
        setup_globals!(V6, 10.)
        x = random_tensors(V6; scale = 0.3)
        cost(; kw...) = real(compute_dimensionless_effective_energy_density(x...; strategy = strategy, μ = 1., Na = 5., kw...))
        commutator = energy_parts(x, 0., 0., 1.).commutator

        @test cost(regulator = 50.) - cost(regulator = 0.) ≈ 50*commutator rtol = 1e-8
        @test cost() ≈ cost(regulator = 200.)                       # the default
    end

    #=
    Number moments used by :number_variance, against direct evaluation:
        ⟨N̂⟩  = L na
        ⟨N̂²⟩ = L ∫₀^L dΔ ⟨:na(Δ) na(0):⟩ + ⟨N̂⟩
    with the integral done by Simpson's rule on dimensionless_density_density_a
    instead of the Van Loan exponential. The ansatz forces Na = Nb, which the
    :number_variance shortcut for equal targets relies on.
    =#
    @testset "number moments match direct integration" begin
        Random.seed!(5)
        setup_globals!(V6, 10.)
        x = random_tensors(V6; scale = 0.3)
        Na_mean, Na²_mean = compute_number_moments(x..., x[2])
        Nb_mean, Nb²_mean = compute_number_moments(x..., x[3])

        @test Na_mean ≈ Lovera*real(density_a(x...)) rtol = 1e-8
        @test Na²_mean - Na_mean^2 > 0                # Var(N̂a)

        Δs = range(0, Lovera, length = 401)
        w = [1; repeat([4, 2], 199); 4; 1]            # Simpson weights
        g = [real(dimensionless_density_density_a(x..., Δ)) for Δ in Δs]
        @test Na²_mean ≈ Lovera*step(Δs)/3*sum(w .* g) + Na_mean rtol = 1e-6

        @test Nb_mean ≈ Na_mean rtol = 1e-8
        @test Nb²_mean ≈ Na²_mean rtol = 1e-8
    end

    #=
    L-BFGS vs gradient descent: from the same start and with the same iteration
    budget, L-BFGS should (1) actually converge and (2) reach a cost at least as
    low as gradient descent. Small setup so it runs quickly: D = 3, L/a = 10 and
    (ga, μ) = (1, 3), which has a non-empty ground state.
    =#
    @testset "L-BFGS converges and beats gradient descent" begin
        Random.seed!(1)
        V3 = U₁Space(-1 => 1, 0 => 1, 1 => 1)
        setup_globals!(V3, 10.)
        x0 = random_tensors(V3)
        cost = x->fg(x...; strategy = :grand_canonical, ga = 1., μ = 3.)

        budget = 1000
        _, f_lbfgs, g_lbfgs, _, _ = optimize(cost, x0, LBFGS(20; verbosity = 0, gradtol = 1e-6, maxiter = budget))
        _, f_gd,    g_gd,    _, _ = optimize(cost, x0, GradientDescent(verbosity = 0, gradtol = 1e-6, maxiter = budget))

        @test norm(g_lbfgs) < 1e-6                  # converged within the budget
        @test f_lbfgs <= f_gd + 1e-10               # at least as low as gradient descent
        println("  after ≤ $(budget) iterations:  L-BFGS |∇f| = $(norm(g_lbfgs)),  GD |∇f| = $(norm(g_gd))")
    end
end
