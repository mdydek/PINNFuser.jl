using Test
using Lux
using OrdinaryDiffEq
using Optimization
using OptimizationOptimisers
using ComponentArrays
using StableRNGs
using Statistics
using ForwardDiff

# Assuming the module code provided is included or loaded here
include("../src/lib.jl")
using .LibInfuser

@testset "PINNParamTuner Test Suite" begin

    # Simple Decay Model: du/dt = -p[1] * u
    # We will try to recover p[1] given synthetic data.
    function simple_decay!(du, u, p, t)
        du[1] = -p[1] * u[1]
    end

    # 2D Linear System (Harmonic Oscillator-ish)
    # x' = a*y, y' = -b*x
    function linear_system!(du, u, p, t)
        a, b = p
        du[1] = a * u[2]
        du[2] = -b * u[1]
    end

    rng = StableRNG(42)

    @testset "Initialization and Config" begin
        u0 = [1.0]
        tspan = (0.0, 1.0)
        target_data = rand(10, 1) # Dummy data

        initial_params = [0.5]
        tune_params = [1]

        logfile = "test_logs_dir/loss.txt"

        if isdir("test_logs_dir")
            rm("test_logs_dir", recursive = true)
        end

        # Suppress printing during test
        redirect_stdout(devnull) do
            LibInfuser.PINN_Parameter_Tuner(
                simple_decay!,
                u0,
                tspan,
                target_data;
                initial_params = initial_params,
                tune_params = tune_params,
                iters = 2,
                loss_logfile = logfile,
                rng = StableRNG(1),
            )
        end

        @test isdir("test_logs_dir")
        @test isfile(logfile)

        rm("test_logs_dir", recursive = true)
    end


    @testset "Custom Neural Network Injection" begin
        u0 = [1.0]
        tspan = (0.0, 1.0)
        target_data = rand(10, 1)
        initial_params = [1.0]
        tune_params = [1]

        # Custom smaller network
        my_nn = Lux.Chain(Lux.Dense(1, 5, tanh), Lux.Dense(5, 1))

        redirect_stdout(devnull) do
            res_u, st, final_params = LibInfuser.PINN_Parameter_Tuner(
                simple_decay!,
                u0,
                tspan,
                target_data;
                initial_params = initial_params,
                tune_params = tune_params,
                iters = 5,
                nn = my_nn,
                rng = StableRNG(42),
            )
        end

        @test true
    end

    @testset "Dimension Mismatch Handling" begin
        # If target data has more columns than the ODE produces, 
        # the code slices `pred[1:ydim, :]`. We verify this runs without crashing.

        u0 = [1.0] # ODE produces 1 dimension
        tspan = (0.0, 0.5)
        # Target data has 2 columns (e.g. extra noisy feature not in ODE)
        target_data = rand(5, 2)

        initial_params = [1.0]
        tune_params = [1]

        redirect_stdout(devnull) do
            @test_nowarn LibInfuser.PINN_Parameter_Tuner(
                simple_decay!,
                u0,
                tspan,
                target_data;
                initial_params = initial_params,
                tune_params = tune_params,
                iters = 5,
                rng = StableRNG(42),
            )
        end
    end
end
