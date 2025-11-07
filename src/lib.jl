module LibInfuser
using Lux, StableRNGs, OptimizationOptimisers, ComponentArrays, LinearAlgebra
using OrdinaryDiffEq, Statistics, ForwardDiff
using SymbolicRegression: SRRegressor
using MLJ: machine, fit!, predict, report

"""
    PINN_Infuser(ode_problem, nn, loss, target_data; alfa=0.1, optimizer=ADAM(), ...)

Trains a Physics-Informed Neural Network (PINN) by minimizing a composite loss function
that includes both data fidelity and physical law adherence.

# Arguments
- `ode_problem::SciMLBase.ODEProblem`: The ODE problem defining the physical laws.
- `nn::Lux.Chain`: The Lux neural network model to be trained.
- `target_data::Array{Float64}`: The ground truth data for training.

# Keyword Arguments
- `alfa::Float64 = 1.0`: The weight factor for the NN infusion in ODE.
- `optimizer = OptimizationOptimisers.Adam`: The optimization algorithm to use.
- `learning_rate::Float64 = 0.001`: The learning rate for the optimizer.
- `iters::Int = 1000`: The number of training iterations.
- `rng::StableRNG` = StableRNG(5958): A random number generator for reproducibility.

# Returns
- `Tuple{Any, Any}`: The trained parameters of the neural network.
"""
function PINN_Infuser(
    ode_problem::SciMLBase.ODEProblem,
    nn::Lux.Chain,
    target_data::AbstractMatrix{Float64};
    alfa::Float64 = 1.0,
    learning_rate::Float64 = 0.001,
    optimizer = Adam,
    iters::Int = 1000,
    rng::StableRNG = StableRNG(5958)
)::Tuple{Any, Any}
    p, st = Lux.setup(rng, nn)
    p = 0.1 * ComponentVector{Float64}(p)

    ode_f = ode_problem.f
    original_p = ode_problem.p
    tsteps = range(ode_problem.tspan[1], ode_problem.tspan[2], length=size(target_data, 1))

    function pinn_ode!(du, u, p, t)
        nn_output = nn(u, p, st)[1]
        ode_f(du, u, original_p, t)
        for i in eachindex(du)
            du[i] *= alfa * (1 + tanh(3.14 * nn_output[i]))
        end
    end

    prob_PINN = ODEProblem(pinn_ode!, ode_problem.u0, ode_problem.tspan, p)

    function predict(p)
        temp_prob = remake(prob_PINN, p=p)
        temp_sol = solve(temp_prob, Tsit5(), saveat=tsteps, reltol=1e-6, abstol=1e-6)
        return temp_sol
    end

    function data_loss(pred, data)
        pred_mat = hcat(pred.u...)'
        return mean(abs2, pred_mat .- data)
    end

    function physics_loss(pred, p)
        pred_mat = hcat(pred.u...)'
        l = 0.0
        
        for (i, t) in enumerate(tsteps)
            u = pred_mat[i, :]
            nn_output = nn(u, p, st)[1]
            
            u_val = ForwardDiff.value.(u)
            du_original = similar(u, Float64)
            ode_f(du_original, u_val, original_p, t)
            
            modulation = alfa .* (1 .+ sin.(3.14 .* nn_output))
            du_modified = du_original .* modulation
            
            l += sum(abs2.(du_modified .- du_original))
        end
        
        return l
    end

    function loss(p)
        pred = predict(p)
        L_data = data_loss(pred, target_data)
        L_phys = physics_loss(pred, p)
        return L_data + 0.5 * L_phys
    end

    adtype = Optimization.AutoForwardDiff()
    optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p)
    losses = Float64[]
    callback = function(p, l)
        push!(losses, l)
        println("Iteration $(length(losses)): Loss = $(losses[end])")
        return false
    end
    trained_params = Optimization.solve(optprob, optimizer(learning_rate), callback=callback, maxiters=iters)
    
    return (trained_params.u, st)
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

function PINN_Parameter_Optimizer()
end


end