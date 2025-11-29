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
- `dtmax::Float64 = 1e-2`: The maximum time step for the ODE solver.
- `iters::Int = 1000`: The number of training iterations.
- `rng::StableRNG` = StableRNG(5958): A random number generator for reproducibility.
- `loss_logfile::String = "training_logs/loss_history.txt"`: File path to log loss history.
- `data_vars::Union{Nothing,Vector{Int}} = nothing`: indices of variables included in data loss. If not defined all variables are used.
- `physics_vars::Union{Nothing,Vector{Int}} = nothing`: indices of variables included in physics loss. If not defined all variables are used.

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
    dtmax = 1e-2,
    iters::Int = 1000,
    rng::StableRNG = StableRNG(5958),
    loss_logfile::String = "training_logs/loss_history.txt",
    data_vars::Union{Nothing,Vector{Int}} = nothing,
    physics_vars::Union{Nothing,Vector{Int}} = nothing,
)::Tuple{Any,Any}
    nvars = length(ode_problem.u0)
    data_vars === nothing && (data_vars = collect(1:nvars))
    physics_vars === nothing && (physics_vars = collect(1:nvars))

    p_init, st = Lux.setup(rng, nn)
    p = ComponentVector{Float64}(p_init) .* 1e-5

    U_MEAN = vec(mean(target_data, dims = 1))
    U_STD = vec(std(target_data, dims = 1)) .+ 1e-6

    # function pinn_ode!(du, u, p, t)
    #     nn_input = (u .- U_MEAN) ./ U_STD
    #     nn_output = nn(nn_input, p, st)[1]
    #     ode_f(du, u, original_p, t)
    #     du .*= 1 .+ (nn_output_weight .* tanh.(nn_output))
    # end

    ode_f = ode_problem.f
    original_p = ode_problem.p
    tsteps =
        range(ode_problem.tspan[1], ode_problem.tspan[2], length = size(target_data, 1))

    function get_nn_output(u, p)
        norm_u = (u .- U_MEAN) ./ U_STD
        return nn(norm_u, p, st)[1]
    end

    function pinn_ode!(du, u, p, t)
        nn_output = get_nn_output(u, p)
        ode_f(du, u, original_p, t)
        du .*= (1 .+ nn_output_weight .* tanh.(nn_output))
    end

    function predict(p)
        temp_prob = ODEProblem(
            (du, u, p, t) -> pinn_ode!(du, u, p, t),
            ode_problem.u0,
            ode_problem.tspan,
            p,
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
        pred_sel = pred_mat[:, data_vars]
        data_sel = data[:, data_vars]
        return mean(abs2, pred_sel .- data_sel)
    end

    function physics_loss(pred, p, physics_vars)
        pred_mat = hcat(pred.u...)'
        l = 0.0

        for (i, t) in enumerate(tsteps)
            u = pred_mat[i, :]

            nn_output = get_nn_output(u, p)
            l += mean(abs2, (tanh.(nn_output[physics_vars])))
        end

        return l
    end

    function loss(p)
        pred = predict(p)
        L_data = data_loss(pred, target_data, data_vars)
        L_phys = physics_loss(pred, p, physics_vars)
        return L_data + physics_weight * L_phys
    end


    adtype = Optimization.AutoForwardDiff()
    optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p)
    losses = Float64[]

    callback = function (p, l)
        push!(losses, l)
        println("Iteration $(length(losses)): Loss = $(losses[end])")

        if early_stopping && length(losses) > 10 && losses[end] - losses[end-10] > 0
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

    # Save loss history
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
