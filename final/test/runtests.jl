using Pkg
Pkg.instantiate()

include("infuser_test.jl")
include("extrapolator_test.jl")
include("regressor_test.jl")
include("parameter_tuner_test.jl")
include("plotter_test.jl")
