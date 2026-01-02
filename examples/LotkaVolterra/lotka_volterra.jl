using DelimitedFiles, StableRNGs, OrdinaryDiffEq, Lux, Plots

include("../../src/lib.jl")
using .LibInfuser

rng = StableRNG(5958)

α = 1.1
β = 0.4
δ = 0.1
γ = 0.4
u0 = [10.0, 5.0]
tspan = (0.0, 20.0)
num_of_samples = 300
tsteps = range(tspan[1], tspan[2], length = num_of_samples)

function lv_with_season!(du, u, p, t)
    x, y = u
    α_season = α * (1.0 + 0.3 * sin(2π / 3))
    du[1] = α_season * x - β * x * y
    du[2] = δ * x * y - γ * y
end

prob_true = ODEProblem(lv_with_season!, u0, tspan)
sol_true = solve(prob_true, Vern7(), saveat = tsteps)

σ_noise = 0.2
data_noisy = sol_true.u .+ σ_noise .* randn.(size.(sol_true.u))
data_noisy_mat = hcat(data_noisy...)'

NN = Lux.Chain(Lux.Dense(2, 20, tanh), Lux.Dense(20, 20, tanh), Lux.Dense(20, 2))

function lv_to_infuse!(du, u, p, t)
    x, y = u
    du[1] = (α * x - β * x * y)
    du[2] = (δ * x * y - γ * y)
end

infusing_problem = ODEProblem(lv_to_infuse!, u0, tspan)

(PINN_solu, trained_st) = LibInfuser.PINN_Infuser(
    infusing_problem,
    NN,
    tsteps,
    data_noisy_mat,
    nn_output_weight = 1.0,
    physics_weight = 1.0,
    learning_rate = 0.01,
    iters = 300,
)

# LibInfuser.PINN_Symbolic_Regressor(
#     NN,
#     (PINN_solu, trained_st)
# )

extrapolation_tspan = (0.0, 60.0)
new_tseps =
    range(extrapolation_tspan[1], extrapolation_tspan[2], length = num_of_samples * 3)

prob_true_extrapolation = ODEProblem(lv_with_season!, u0, extrapolation_tspan)
sol_true_extrapolation = solve(prob_true_extrapolation, Vern7(), saveat = new_tseps)
u_true_mat = hcat(sol_true_extrapolation.u...)'

LibInfuser.PINN_Extrapolator(
    infusing_problem,
    NN,
    (PINN_solu, trained_st),
    extrapolation_tspan,
    num_of_samples * 3,
    "lotka_extrapolation.csv",
    nn_output_weight = 1.0,
)

pred_mat = readdlm("lotka_extrapolation.csv", ',', Float64)

prob_ODE = ODEProblem(
    (du, u, p, t) ->
        (du[1] = α * u[1] - β * u[1] * u[2]; du[2] = δ * u[1] * u[2] - γ * u[2]),
    u0,
    extrapolation_tspan,
)
sol_ODE = solve(prob_ODE, Vern7(), saveat = new_tseps)
ode_solution = hcat(sol_ODE.u...)'

# --- Prey ---
LibInfuser.PINNPlotter.plot_PINN_results(
    pred_mat[:, 1:1],
    data_noisy_mat[:, 1:1],
    ode_solution[:, 1:1],
    ["Prey data", "PINN Prey", "ODE Prey (no NN)"],
    tsteps,
    new_tseps,
    "t",
    "Population",
    "Lotka-Volterra: Prey (Only)",
    "lotka_prey_plot.png",
)

# --- Predator ---
LibInfuser.PINNPlotter.plot_PINN_results(
    pred_mat[:, 2:2],
    data_noisy_mat[:, 2:2],
    ode_solution[:, 2:2],
    ["Predator data", "PINN Predator", "ODE Predator (no NN)"],
    tsteps,
    new_tseps,
    "t",
    "Population",
    "Lotka-Volterra: Predator (Only)",
    "lotka_predator_plot.png",
)

LibInfuser.PINNPlotter.plot_loss("training_logs/loss_history.txt")

# plot(new_tseps, u_true_mat[:, 1], label = "Prey ground truth", lw = 2, ls = :dot)
# plot!(new_tseps, pred_mat[:, 1], label = "PINN Prey", lw = 3)
# plot!(new_tseps, sol_ODE[1, :], label = "ODE Prey (no NN)", lw = 2, ls = :dash)
# plot(new_tseps, u_true_mat[:, 2], label = "Predator ground truth", lw = 2, ls = :dot)
# plot!(new_tseps, pred_mat[:, 2], label = "PINN Predator", lw = 3)
# plot!(new_tseps, sol_ODE[2, :], label = "ODE Predator (no NN)", lw = 2, ls = :dash)
# xlabel!("t")
# ylabel!("Population")
# title!("Lotka-Volterra: PINN vs Noisy Data vs Ideal ODE extrapolation")
# savefig("experiments/lotka_volterra_plots/lib_extrapolation.png")
# println("Plot saved as lib_extrapolation.png")
