using Test
using Lux
using OrdinaryDiffEq
using DelimitedFiles
using Statistics
using Random

# Assuming the module code provided is included or loaded here
include("../lib/lib.jl")
using .LibInfuser

@testset "PINN_Extrapolator Test Suite" begin

    # --- Test Fixtures and Setup ---
    rng = Random.default_rng()
    Random.seed!(rng, 42)

    # 1. Define a simple Base ODE (Exponential decay: du/dt = -u)
    function simple_decay!(du, u, p, t)
        du[1] = -u[1]
    end

    u0 = [10.0]
    tspan = (0.0, 1.0)
    base_prob = ODEProblem(simple_decay!, u0, tspan)

    # 2. Define a simple Lux Neural Network
    nn = Chain(Dense(1 => 5, tanh), Dense(5 => 1))
    ps, st = Lux.setup(rng, nn)
    pretrained_params = (ps, st)

    # 3. Create Dummy Target Data (Rows = samples, Cols = features)
    # Needed for Mean/Std normalization logic inside the function
    target_data = rand(rng, 100, 1) .* 10

    # 4. File path configuration
    test_output_path = "test_prediction_output.csv"

    @testset "File I/O and Output Shape" begin
        num_samples = 20
        alpha = 0.1

        if isfile(test_output_path)
            rm(test_output_path)
        end

        LibInfuser.PINN_Extrapolator(
            base_prob,
            tspan,
            target_data,
            alpha,
            num_samples,
            nn,
            pretrained_params,
            test_output_path,
        )

        @test isfile(test_output_path)

        data = readdlm(test_output_path, ',')

        @test size(data) == (num_samples, length(u0))
        @test eltype(data) <: AbstractFloat

        rm(test_output_path)
    end

    @testset "Physics Consistency (Alpha = 0)" begin
        # When alpha is 0.0, the Neural Network contribution should be nullified.
        # The result should match the standard ODE solver exactly (within tolerance).

        num_samples = 50
        alpha = 0.0

        LibInfuser.PINN_Extrapolator(
            base_prob,
            tspan,
            target_data,
            alpha,
            num_samples,
            nn,
            pretrained_params,
            test_output_path,
        )

        pinn_result = readdlm(test_output_path, ',')

        save_times = range(tspan[1], tspan[2], length = num_samples)
        standard_sol =
            solve(base_prob, Vern7(), saveat = save_times, reltol = 1e-6, abstol = 1e-6)

        standard_result = hcat(standard_sol.u...)'

        @test isapprox(pinn_result, standard_result, atol = 1e-5)

        rm(test_output_path)
    end

    @testset "System Integration (Multi-dimensional)" begin
        # Test with a 2D ODE system to ensure dimensions are handled correctly
        # Harmonic oscillator: x' = v, v' = -x
        function harmonic!(du, u, p, t)
            du[1] = u[2]
            du[2] = -u[1]
        end

        u0_2d = [0.0, 1.0]
        prob_2d = ODEProblem(harmonic!, u0_2d, tspan)

        # NN input dim must match state dim (2)
        nn_2d = Chain(Dense(2 => 5), Dense(5 => 2))
        ps_2d, st_2d = Lux.setup(rng, nn_2d)

        target_data_2d = rand(rng, 50, 2)
        num_samples = 15
        alpha = 0.5

        LibInfuser.PINN_Extrapolator(
            prob_2d,
            tspan,
            target_data_2d,
            alpha,
            num_samples,
            nn_2d,
            (ps_2d, st_2d),
            test_output_path,
        )

        data_2d = readdlm(test_output_path, ',')

        @test size(data_2d) == (num_samples, 2)
        @test !any(isnan, data_2d)

        rm(test_output_path)
    end
end
