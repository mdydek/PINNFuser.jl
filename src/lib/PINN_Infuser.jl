using Lux, StableRNGs, OptimizationOptimisers, ComponentArrays, LinearAlgebra
using OrdinaryDiffEq, Statistics, ForwardDiff

"""
    PINN_Infuser(ode_problem, nn, loss, target_data; alfa=0.1, optimizer=ADAM(), ...)

Trains a Physics-Informed Neural Network (PINN) by minimizing a composite loss function
that includes both data fidelity and physical law adherence.

# Arguments
- `ode_problem::SciMLBase.ODEProblem`: The ODE problem defining the physical laws.
- `nn::Lux.Chain`: The Lux neural network model to be trained.
- `target_data::Array{Float64}`: The ground truth data for training.

# Keyword Arguments
- `early_stopping::Bool = true`: Whether to enable early stopping based on loss convergence.
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
    early_stopping::Bool = true,
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
        return L_data + 2 * L_phys
    end

    adtype = Optimization.AutoForwardDiff()
    optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p)
    losses = Float64[]
    callback = function(p, l)
        push!(losses, l)
        println("Iteration $(length(losses)): Loss = $(losses[end])")
        if early_stopping && length(losses) > 10 && abs(losses[end] - losses[end-10]) < 1e-3
            println("Early stopping at iteration $(length(losses)) with loss $(losses[end])")
            return true
        else 
            return false
        end
    end
    trained_params = Optimization.solve(optprob, optimizer(learning_rate), callback=callback, maxiters=iters)
    
    return (trained_params.u, st)
end