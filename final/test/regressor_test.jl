using Test
using Lux
using MLJ
using SymbolicRegression
using StableRNGs

include("../lib/lib.jl")
using .LibInfuser

function capture_stdout(f)
    pipe = Pipe()

    redirect_stdout(pipe) do
        f()
    end

    close(Base.pipe_writer(pipe))
    return read(pipe, String)
end
# --------------------------------------------

@testset "PINN_Symbolic_Regressor Test Suite" begin

    @testset "Integration Test with Simple Function" begin
        # Setup a simple deterministic Neural Network
        nn = Chain(Dense(1 => 1, identity))
        rng = StableRNG(42)
        ps, st = Lux.setup(rng, nn)

        # Manually set weights to 2.0 and bias to 1.0
        ps_fixed = (layer_1 = (weight = [2.0;;], bias = [1.0]),)
        pretrained_params = (ps_fixed, st)

        output_str = capture_stdout() do
            LibInfuser.PINN_Symbolic_Regressor(nn, pretrained_params, 100)
        end

        @test occursin("Finding equation", output_str)
    end

    @testset "Multi-dimensional Output Handling" begin
        # Test if it handles a network with 2 output dimensions correctly
        nn_multi = Chain(Dense(2 => 2, identity))
        rng = StableRNG(123)
        ps, st = Lux.setup(rng, nn_multi)

        pretrained_params = (ps, st)

        output_str = capture_stdout() do
            LibInfuser.PINN_Symbolic_Regressor(nn_multi, pretrained_params, 100)
        end

        @test occursin("Finding equation for Output 1", output_str)
        @test occursin("Finding equation for Output 2", output_str)
    end
end
