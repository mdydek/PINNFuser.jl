module LibInfuser
using Lux, SciMLBase, StableRNGs, OptimizationOptimisers, ComponentArrays

"""
    PINN_Infuser(ode_problem, nn, loss, target_data; alfa=0.1, optimizer=ADAM(), ...)

Trains a Physics-Informed Neural Network (PINN) by minimizing a composite loss function
that includes both data fidelity and physical law adherence.

# Arguments
- `ode_problem::SciMLBase.ODEProblem`: The ODE problem defining the physical laws.
- `nn::Lux.Chain`: The Lux neural network model to be trained.
- `loss`: A function `(predicted, true) -> value` that computes the data loss.
- `target_data::Array{Float64}`: The ground truth data for training.

# Keyword Arguments
- `alfa::Float64 = 0.1`: The weight factor for the NN infusion in ODE.
- `optimizer = OptimizationOptimisers.ADAM()`: The optimization algorithm to use.
- `learning_rate::Float64 = 0.001`: The learning rate for the optimizer.
- `iters::Int = 1000`: The number of training iterations.
- `rng::StableRNG`: A random number generator for reproducibility.

# Returns
- `ComponentArray{Float64}`: The trained parameters of the neural network.
"""
function PINN_Infuser(
    ode_problem::SciMLBase.ODEProblem, 
    nn::Lux.Chain, 
    loss::(predicted_data::Array{Float64}, true_data::Array{Float64}) -> Float64,
    target_data::Array{Float64},
    alfa::Float64 = 0.1,
    optimizer::OptimizationOptimisers.OptimizationFunction = OptimizationOptimisers.ADAM(),
    learning_rate::Float64 = 0.001,
    iters::Int = 1000,
    rng::StableRNG = StableRNG(5958)
)::ComponentArray{Float64}

end

"""
    PINN_Extrapolator(ode_problem, tspan, num_of_samples, u0, pretrained_params, path_to_save)

Solves a PINN-infused ODE problem to extrapolate its trajectory and saves the result to a file.

This function takes a pre-trained neural network's parameters (`pretrained_params`) and an `ODEProblem` that uses them. It then solves this problem over a new time span `tspan` starting from a new initial condition `u0`, generating a specified number of sample points. The resulting time series data is saved to `path_to_save`.

# Arguments
- `ode_problem::SciMLBase.ODEProblem`: The ODE problem defining the system's dynamics, which should be parameterized by the neural network.
- `tspan::Tuple{Float64, Float64}`: The time interval `(t_start, t_end)` for the extrapolation.
- `num_of_samples::Int`: The number of evenly spaced time points to generate and save within the `tspan`.
- `alfa::Float64`: The weight factor for the NN infusion in ODE.
- `u0::Array{Float64}`: The initial condition (state vector) from which to start the extrapolation.
- `nn::Lux.Chain`: The Lux neural network model structure.
- `pretrained_params::ComponentArray{Float64}`: The trained parameters of the neural network.
- `path_to_save::String`: The full file path (e.g., `"data/prediction.csv"`) where the output will be saved.

# Returns
- The function is intended for side-effects (saving a file) and may not have a meaningful return value (`nothing`).
"""
function PINN_Extrapolator(
    ode_problem::SciMLBase.ODEProblem,
    tspan::Tuple{Float64, Float64},
    alfa::Float64,
    num_of_samples::Int,
    u0::Array{Float64},
    nn::Lux.Chain,
    pretrained_params::ComponentArray{Float64},
    path_to_save::String
)::Nothing
end

"""
    PINN_Symbolic_Regressor(ode_problem, nn, pretrained_params)
Wraps a pre-trained neural network into an ODE problem for symbolic regression.
# Arguments
- `ode_problem::SciMLBase.ODEProblem`: The original ODE problem defining the
system's dynamics.
- `nn::Lux.Chain`: The Lux neural network model structure.
- `pretrained_params::ComponentArray{Float64}`: The trained parameters of the neural
network.
# Returns
- `SciMLBase.ODEProblem`: A new ODE problem without the neural network infused,
but incorporating the learned behavior via symbolic regression. This ODE problem can be
used for further analysis or simulation.
"""
function PINN_Symbolic_Regressor(
    ode_problem::SciMLBase.ODEProblem,
    nn::Lux.Chain,
    pretrained_params::ComponentArray{Float64},
)::SciMLBase.ODEProblem
end


end