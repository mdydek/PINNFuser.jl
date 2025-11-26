using DelimitedFiles, StableRNGs, OrdinaryDiffEq, Lux

rng = StableRNG(5958)
include("../src/lib/lib.jl")

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
    α_season = α * (1.0 + 0.3 * sin(2π / 10))
    du[1] = α_season * x - β * x * y
    du[2] = δ * x * y - γ * y
end

prob_true = ODEProblem(lv_with_season!, u0, tspan)
sol_true = solve(prob_true, Tsit5(), saveat = tsteps)
σ_noise = 0.2
data_noisy = sol_true.u .+ σ_noise .* randn.(size.(sol_true.u))
data_noisy_mat = hcat(data_noisy...)'

NN = Lux.Chain(Lux.Dense(2, 20, elu), Lux.Dense(20, 20, elu), Lux.Dense(20, 2))

function lv_to_infuse!(du, u, p, t)
    x, y = u
    du[1] = (α * x - β * x * y)
    du[2] = (δ * x * y - γ * y)
end

infusing_problem = ODEProblem(lv_to_infuse!, u0, tspan)

(PINN_solu, trained_st) =
    LibInfuser.PINN_Infuser(infusing_problem, NN, data_noisy_mat, iters = 250)

# LibInfuser.PINN_Symbolic_Regressor(
#     NN,
#     (PINN_solu, trained_st)
# )

extrapolation_tspan = (0.0, 60.0)
new_tseps =
    range(extrapolation_tspan[1], extrapolation_tspan[2], length = num_of_samples * 3)

# Get true solution
prob_true_extrapolation = ODEProblem(lv_with_season!, u0, extrapolation_tspan)
sol_true_extrapolation = solve(prob_true_extrapolation, Tsit5(), saveat = new_tseps)
u_true_mat = hcat(sol_true_extrapolation.u...)'

# Get PINN prediction

LibInfuser.PINN_Extrapolator(
    infusing_problem,
    extrapolation_tspan,
    0.1,
    num_of_samples * 3,
    NN,
    (PINN_solu, trained_st),
    "lotka_extrapolation.csv",
)

pred_mat = readdlm("lotka_extrapolation.csv", ',', Float64)

# Get standard ODE solution
prob_ODE = ODEProblem(
    (du, u, p, t) ->
        (du[1] = α * u[1] - β * u[1] * u[2]; du[2] = δ * u[1] * u[2] - γ * u[2]),
    u0,
    extrapolation_tspan,
)
sol_ODE = solve(prob_ODE, Tsit5(), saveat = new_tseps)
ode_solution = hcat(sol_ODE.u...)'

LibInfuser.PINN_Plotter(
    pred_mat,
    data_noisy_mat,
    ode_solution,
    [
        "Prey data",
        "PINN Prey",
        "ODE Prey (no NN)",
        "Predator data",
        "PINN Predator",
        "ODE Predator (no NN)",
    ],
    tsteps,
    new_tseps,
    "t",
    "Population",
    "Lotka-Volterra: Data vs PINN vs simple ODE",
    "lotka_pinn_plot.png",
)
