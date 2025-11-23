module PINNExtrapolator

using Lux, DelimitedFiles, OrdinaryDiffEq

export PINN_Extrapolator

"""
    PINN_Extrapolator(ode_problem, tspan, num_of_samples, u0, pretrained_params, path_to_save)

Solves a PINN-infused ODE problem to extrapolate its trajectory and saves the result to a file.

This function takes a pre-trained neural network's parameters (`pretrained_params`) and an `ODEProblem` that uses them. It then solves this problem over a new time span `tspan` starting from a new initial condition `u0`, generating a specified number of sample points. The resulting time series data is saved to `path_to_save`.

# Arguments
- `base_problem::SciMLBase.ODEProblem`: Base ODE problem.
- `tspan::Tuple{Float64, Float64}`: The time interval `(t_start, t_end)` for the extrapolation.
- `num_of_samples::Int`: The number of evenly spaced time points to generate and save within the `tspan`.
- `alfa::Float64`: The weight factor for the NN infusion in ODE.
- `nn::Lux.Chain`: The Lux neural network model structure.
- `pretrained_params::Tuple{Any, Any}`: The trained parameters of the neural network.
- `path_to_save::String`: The full file path (e.g., `"data/prediction.csv"`) where the output will be saved.
"""
function PINN_Extrapolator(
    base_problem::SciMLBase.ODEProblem,
    tspan::Tuple{Float64, Float64},
    alpha::Float64,
    num_of_samples::Int,
    nn::Lux.Chain,
    pretrained_params::Tuple{Any, Any},
    path_to_save::String
)::Nothing
    trained_u, trained_st = pretrained_params
    new_tseps = range(tspan[1], tspan[2], length=num_of_samples)

    function pinn_ode!(du, u, p, t)
        nn_output = nn(u, trained_u, trained_st)[1]
        base_problem.f(du, u, base_problem.p, t)
        du .*= 1 .+ (alpha .* sin.(nn_output))
    end

    pinn_problem = ODEProblem(pinn_ode!, base_problem.u0, tspan)

    solved_pinn = solve(pinn_problem, Tsit5(), saveat=new_tseps, reltol=1e-6, abstol=1e-6)

    pred_mat = hcat(solved_pinn.u...)'

    writedlm(path_to_save, pred_mat, ',')
end

end # module PINNExtrapolator