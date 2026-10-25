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
    return tr(tmp*exp(Lovera*TM))/compute_norm(F, Ja⁺, Jb⁺)
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
        n = compute_a_particle_density(x...)

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

    @testset "gradient matches finite difference ($(kw.strategy))" for kw in [
            (strategy = :grand_canonical, ga = 2.5, μ = 5.),
            (strategy = :fixed_density,   ga = 2.5, N = 0.5, λ = 10.)]
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

    #=
    The two strategies should give the same ground state.

    At the fixed-density optimum the penalty acts as μ_eff = 2λ(N - n). So if the
    grand-canonical run at μ gives density n_gc, the fixed-density run with
        N = n_gc + μ/(2λ)
    has its minimum at exactly the same state (not only for large λ).

    Small setup so it runs quickly: D = 3, L/a = 10, and (ga, μ) = (1, 3), which
    has a non-empty ground state (n ≈ 0.06). Both runs start from the same random
    tensors, so the fixed-density run has to find the state on its own.
    =#
    @testset "grand canonical and fixed density agree" begin
        Random.seed!(1)
        V3 = U₁Space(-1 => 1, 0 => 1, 1 => 1)
        setup_globals!(V3, 10.)
        x0 = random_tensors(V3)

        ga, μ, λ = 1., 3., 10.
        alg = LBFGS(20; verbosity = 0, gradtol = 1e-8, maxiter = 3000)

        x_gc, _ = optimize(x->fg(x...; strategy = :grand_canonical, ga = ga, μ = μ), x0, alg)
        n_gc = real(compute_a_particle_density(x_gc...))
        E_gc = real(compute_a_squared_energy_density(x_gc...; ga = ga))

        # make sure we're not comparing two empty (vacuum) states
        @test n_gc > 0.01

        N = n_gc + μ/(2λ)
        x_fd, _ = optimize(x->fg(x...; strategy = :fixed_density, ga = ga, N = N, λ = λ), x0, alg)
        n_fd = real(compute_a_particle_density(x_fd...))
        E_fd = real(compute_a_squared_energy_density(x_fd...; ga = ga))

        @test n_fd ≈ n_gc rtol = 1e-6
        @test E_fd ≈ E_gc rtol = 1e-6
        @test 2λ*(N - n_fd) ≈ μ rtol = 1e-6       # recovered chemical potential

        # the tensors themselves are only fixed up to a gauge, so compare observables
        for Δx in (0.5, 2., 5.)
            @test dimensionless_two_point_function_a_a_times_L(x_fd..., Δx) ≈
                  dimensionless_two_point_function_a_a_times_L(x_gc..., Δx) atol = 1e-8
        end
    end

    #=
    L-BFGS vs gradient descent: from the same start and with the same iteration
    budget, L-BFGS should (1) actually converge and (2) reach a cost at least as
    low as gradient descent. Same small setup as the test above.
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
