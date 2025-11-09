using Lux, StableRNGs
using SymbolicRegression: SRRegressor
using MLJ: machine, fit!, predict, report

"""
    PINN_Symbolic_Regressor(ode_problem, nn, pretrained_params)
Wraps a pre-trained neural network into an ODE problem for symbolic regression.
# Arguments
- `nn::Lux.Chain`: The Lux neural network model structure.
- `pretrained_params::Tuple{Any, Any}`: The trained parameters of the neural
network.
"""
function PINN_Symbolic_Regressor(
    nn::Lux.Chain,
    pretrained_params::Tuple{Any, Any},
)

    trained_u, trained_st = pretrained_params
    num_samples = 500
    in_dims = nn[1].in_dims
    X_lux = 100 * rand(Float64, in_dims, num_samples)

    y_matrix = nn(X_lux, trained_u, trained_st)[1]

    X_mlj = transpose(X_lux)
    
    out_dims = size(y_matrix, 1)
        
    sr_model = SRRegressor(
        niterations=500,
        binary_operators=[+, -, *, /],
        unary_operators=[cos, exp, sin],
        maxsize=15
    )

    for j in 1:out_dims
        println("\n--- Finding equation for Output $j ---")
        
        y_j_mlj = vec(y_matrix[j, :])
        
        mach = machine(sr_model, X_mlj, y_j_mlj)
        fit!(mach)
        
        report(mach)
    end
end