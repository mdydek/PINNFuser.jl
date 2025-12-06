module PINNInfuser

using Lux, StableRNGs, Optimization, OptimizationOptimisers, ComponentArrays, LinearAlgebra
using OrdinaryDiffEq, Statistics, ForwardDiff
using Printf

export PINN_Infuser

"""
    PINN_Infuser(ode_problem, nn, loss, target_data; nn_output_weight=0.1, physics_weight = 1.0, optimizer=ADAM(), ...)

Trains a Physics-Informed Neural Network (PINN) by minimizing a composite loss function
that includes both data fidelity and physical law adherence.

# Arguments
- `ode_problem::SciMLBase.ODEProblem`: The ODE problem defining the physical laws.
- `nn::Lux.Chain`: The Lux neural network model to be trained.
- `target_data::Array{Float64}`: The ground truth data for training.

# Keyword Arguments
- `early_stopping::Bool = true`: Whether to enable early stopping based on loss convergence.
- `nn_output_weight::Float64 = 0.1`: The weight factor for the NN infusion in ODE.
- `physics_weight::Float64 = 1.0`: The weight of the physics-based loss component.
- `optimizer = OptimizationOptimisers.Adam`: The optimization algorithm to use.
- `learning_rate::Float64 = 0.001`: The learning rate for the optimizer.
- `reltol::Float64 = 1e-6`: The relative tolerance for the ODE solver.
- `abstol::Float64 = 1e-6`: The absolute tolerance for the ODE solver.
- `dtmax = Inf`: The maximum time step for the ODE solver.
- `iters::Int = 1000`: The number of training iterations.
- `rng::StableRNG` = StableRNG(5958): A random number generator for reproducibility.
- `loss_logfile::String = "training_logs/loss_history.txt"`: File path to log loss history.
- `data_vars::Union{Nothing,Vector{Int}} = nothing`: Indices of variables to include in data loss.
- `physics_vars::Union{Nothing,Vector{Int}} = nothing`: Indices of variables to include in physics loss.

# Returns
- `Tuple{Any, Any}`: The trained parameters of the neural network.
"""
function PINN_Infuser(
    ode_problem::SciMLBase.ODEProblem,
    nn::Lux.Chain,
    target_data::AbstractMatrix{Float64};
    early_stopping::Bool = true,
    nn_output_weight::Float64 = 0.1,
    physics_weight::Float64 = 1.0,
    optimizer = ADAM,
    learning_rate::Float64 = 0.001,
    reltol::Float64 = 1e-6,
    abstol::Float64 = 1e-6,
    dtmax = Inf,
    iters::Int = 1000,
    rng::StableRNG = StableRNG(5958),
    loss_logfile::String = "training_logs/loss_history.txt",
    data_vars::Union{Nothing,Vector{Int}} = nothing,
    physics_vars::Union{Nothing,Vector{Int}} = nothing,
)::Tuple{Any,Any}
    nvars = length(ode_problem.u0)
    data_vars === nothing && (data_vars = collect(1:nvars))
    physics_vars === nothing && (physics_vars = collect(1:nvars))

    p_NN, st = Lux.setup(rng, nn)
    p_NN = 0 * ComponentVector{Float64}(p_NN)

    ode_f = ode_problem.f
    tsteps = range(ode_problem.tspan[1], ode_problem.tspan[2], length = size(target_data, 1))

    function pinn_ode!(du, u, p_NN, t)
        nn_output = nn(u, p_NN, st)[1]
        ode_f(du, u, nothing, t)
        du .*= 1 .+ nn_output_weight .* tanh.(nn_output)
    end

    function predict(p_NN)
        temp_prob = ODEProblem(
            (du, u, p_NN, t) -> pinn_ode!(du, u, p_NN, t),
            ode_problem.u0,
            ode_problem.tspan,
            p_NN,
        )
        temp_sol = solve(
            temp_prob,
            Vern7(),
            saveat = tsteps,
            dtmax = dtmax,
            reltol = reltol,
            abstol = abstol,
        )
        return temp_sol
    end

    function data_loss(pred, data, data_vars)
        pred_mat = hcat(pred.u...)'
        return sum(mean(abs2, pred_mat[:, j] .- data[:, j]) for j in data_vars)
    end

    function physics_loss(pred, p_NN, physics_vars)
        pred_mat = hcat(pred.u...)'
        l = 0.0
        for (i, t) in enumerate(tsteps)
            u = pred_mat[i, :]
            du = similar(u)
            pinn_ode!(du, u, p_NN, t)
            f_base = similar(u)
            ode_f(f_base, u, nothing, t)
            l += mean(abs2.(du[physics_vars] .- f_base[physics_vars]))
        end
        return l / length(tsteps)
    end

    function loss(p_NN)
        pred = predict(p_NN)
        L_data = data_loss(pred, target_data, data_vars)
        L_phys = physics_loss(pred, p_NN, physics_vars)
        return L_data + physics_weight * L_phys
    end

    adtype = Optimization.AutoForwardDiff()
    optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p_NN)
    losses = Float64[]

    callback = function (p_NN, l)
        push!(losses, l)
        println("Iteration $(length(losses)): Loss = $(losses[end])")

        if early_stopping && length(losses) > 100 && losses[end] - losses[end-10] > 0
            println("Early stopping at iteration $(length(losses)) with loss $(losses[end])")
            return true
        else
            return false
        end
    end

    trained_params = Optimization.solve(
        optprob,
        optimizer(learning_rate),
        callback = callback,
        maxiters = iters,
    )

    folder = dirname(loss_logfile)
    if folder != "" && !isdir(folder)
        println("Creating directory for training logs: $folder")
        mkpath(folder)
    end

    open(loss_logfile, "w") do io
        for (i, L) in enumerate(losses)
            @printf(io, "%d %.12f\n", i, L)
        end
    end

    return (trained_params.u, st)
end

end # module PINNInfuser
