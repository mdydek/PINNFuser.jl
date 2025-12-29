using Test
using Lux,
    OrdinaryDiffEq,
    StableRNGs,
    Statistics,
    OptimizationOptimisers,
    LinearAlgebra,
    ComponentArrays

include("../lib/lib.jl")
using .LibInfuser

@testset "PINNInfuser Tests" begin

    function oscillator(du, u, p, t)
        du[1] = u[2]
        du[2] = -p[1] * u[1]
    end

    u0 = [1.0, 0.0]
    tspan = (0.0, 5.0)
    p_true = [1.0]
    prob = ODEProblem(oscillator, u0, tspan, p_true)

    function damped_oscillator(du, u, p, t)
        du[1] = u[2]
        du[2] = -p[1] * u[1] - 0.1 * u[2]
    end
    prob_target = ODEProblem(damped_oscillator, u0, tspan, p_true)
    sol_target = solve(prob_target, Vern7(), saveat = 0.1)
    target_data = hcat(sol_target.u...)' |> Matrix{Float64}
    tsteps = range(tspan[1], tspan[2], length = size(target_data, 1))

    rng = StableRNG(42)
    nn = Chain(Dense(2, 10, tanh), Dense(10, 2))

    @testset "Init test" begin
        params, st = LibInfuser.PINN_Infuser(
            prob,
            nn,
            tsteps,
            target_data,
            iters = 10,
            optimizer = ADAM,
            loss_logfile = "test_logs/smoke_test.txt",
        )

        @test params isa ComponentVector || params isa AbstractVector
        @test st isa NamedTuple
        @test isfile("test_logs/smoke_test.txt")
        rm("test_logs/smoke_test.txt", force = true)
    end

    @testset "Convergence test" begin

        logfile = "test_logs/conv_test.txt"
        LibInfuser.PINN_Infuser(
            prob,
            nn,
            tsteps,
            target_data,
            iters = 100,
            learning_rate = 0.01,
            loss_logfile = logfile,
            early_stopping = false,
        )

        lines = args = readlines(logfile)
        losses = [parse(Float64, split(l)[2]) for l in lines]

        first_loss = losses[1]
        last_loss = losses[end-1]

        @test last_loss < first_loss
        println("Loss reduction: $first_loss -> $last_loss")
        rm(logfile, force = true)
    end

    @testset "Variable Selection test" begin

        data_vars = [1]
        physics_vars = [1, 2]

        params, st = LibInfuser.PINN_Infuser(
            prob,
            nn,
            tsteps,
            target_data,
            iters = 10,
            data_vars = data_vars,
            physics_vars = physics_vars,
            loss_logfile = "test_logs/partial_vars.txt",
        )

        @test params !== nothing
        rm("test_logs/partial_vars.txt", force = true)
        rm("test_logs", recursive = true, force = true)
    end

    @testset "Deterministic test" begin
        rng1 = StableRNG(123)
        res1, _ =
            LibInfuser.PINN_Infuser(prob, nn, tsteps, target_data, rng = rng1, iters = 5)

        rng2 = StableRNG(123)
        res2, _ =
            LibInfuser.PINN_Infuser(prob, nn, tsteps, target_data, rng = rng2, iters = 5)

        @test res1 == res2
    end
end
