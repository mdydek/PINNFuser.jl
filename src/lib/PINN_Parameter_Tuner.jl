module PINNParamTuner

using OrdinaryDiffEq
using Lux
using ComponentArrays
using Optimization
using OptimizationOptimisers
using Statistics

export PINN_Parameter_Tuner


"""
    PINN_Parameter_Tuner(model_ode!, u0, tspan, target_data; initial_params, tune_params, ...)

Trains a neural network to tune selected constant parameters of an ODE model.

# Arguments
- `model_ode!(du, u, p, t)::Function`: The user-defined ODE function, expecting a parameter vector `p`.
- `u0::AbstractVector{<:Real}`: Initial state vector of the system.
- `tspan::Tuple`: Time span of the simulation, e.g., `(0.0, 7.0)`.
- `target_data::AbstractMatrix`: Reference data used to train the network.

# Keyword Arguments
- `initial_params::Vector{<:Real}`: Initial values for the model parameters.
- `tune_params::Vector{Int}`: Indices of parameters to be tuned by the neural network.
- `range_fraction::Float64 = 0.6`: Allowed relative range for each tunable parameter.
- `learning_rate::Float64 = 0.01`: Learning rate for the optimizer.
- `optimizer = ADAM`: Optimization algorithm (e.g., ADAM, RMSProp).
- `iters::Int = 1000`: Number of training iterations.
- `rng = StableRNG(1234)`: Random number generator for reproducibility.
- `nn::Union{Nothing, Lux.Chain} = nothing`: Optional user-provided neural network. If `nothing`, a default network is created.

# Returns
- `res.u::ComponentVector{Float64}`: Trained neural network parameters.
- `st::NamedTuple`: Network state (required for predictions with Lux).
- `tuned_list::Vector{Vector{Float64}}`: List of parameter vectors generated for the first `num_param_samples` states in `target_data`.
"""

function PINN_Parameter_Tuner(
    model_ode!::Function,
    u0::AbstractVector{<:Real},
    tspan::Tuple,
    target_data::AbstractMatrix;
    initial_params::Vector{<:Real},
    tune_params::Vector{Int},
    range_fraction::Float64 = 0.6,
    learning_rate::Float64 = 0.01,
    optimizer = ADAM,
    iters::Int = 1000,
    rng = StableRNG(1234),
    nn::Union{Nothing, Lux.Chain} = nothing
)

    nin = length(u0)
    nout = length(tune_params)

    # Default NN if none provided
    if nn === nothing
        nn = Lux.Chain(
            Lux.Dense(nin, 32, tanh),
            Lux.Dense(32, 32, tanh),
            Lux.Dense(32, nout)
        )
    end

    nn_params, st = Lux.setup(rng, nn)
    nn_params = 0 * ComponentVector{Float64}(nn_params)
    
    tsteps = range(tspan[1], tspan[2], length=size(target_data,1))

    function param_from_NN(u, p, st)
        y, _ = nn(u, p, st)
        pars = similar(y, length(initial_params))

        for (k, idx) in enumerate(tune_params)
            base = initial_params[idx]
            minv = base * (1 - range_fraction)
            maxv = base * (1 + range_fraction)
            pars[idx] = minv + abs(y[k]) * (maxv - minv)
        end

        for i in 1:length(initial_params)
            if !(i in tune_params)
                pars[i] = initial_params[i]
            end
        end

        return pars
    end

    function wrapped_ODE!(du, u, p, t)
        p_new = param_from_NN(u, p, st)
        model_ode!(du, u, p_new, t)
    end

    function predict(nn_p)
        prob = ODEProblem(wrapped_ODE!, u0, tspan, nn_p)
        solve(prob, Vern7(), dtmax=1e-2, saveat=tsteps, reltol=1e-6, abstol=1e-6)
    end

    function loss(p)
        sol = predict(p)
        pred = Array(sol)
        ydim = min(size(pred,1), size(target_data,2))

        pred_slice = pred[1:ydim, :]'
        true_slice = target_data[:, 1:ydim]

        mean(abs2, pred_slice .- true_slice)
    end

    function make_callback()
        iter = Ref(0)
        return function (p, l)
            if iter[] % 1 == 0
                println("iter $(iter[]) | loss = $(round(l, sigdigits=5))")
            end
            iter[] += 1
            return false
        end
    end
    cb = make_callback()

    adtype = Optimization.AutoForwardDiff()
    optf = Optimization.OptimizationFunction((x,p)->loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, nn_params)

    res = Optimization.solve(optprob, optimizer(learning_rate),
                             callback=cb, maxiters=iters)

    final_params = param_from_NN(target_data[1, 1:length(u0)], res.u, st)

    return res.u, st, final_params
end

end