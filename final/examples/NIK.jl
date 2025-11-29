using OrdinaryDiffEq
using Lux, Plots, Zygote, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Optim, Measures, BenchmarkTools
using DelimitedFiles, ForwardDiff

include("../models/simple_CVS_model.jl")
using .simpleModel

include("../lib/lib.jl")
using .LibInfuser

rng = StableRNG(5958)

# Training range
tspan = (0.0, 7.0)
num_of_samples = 300
tsteps = range(5.0, 7.0, length = num_of_samples)

loaded_data = readdlm("src/data/original_data.txt")
original_data = Array{Float64}(loaded_data)
start_idx = 751
original_data = original_data[751:1050, :]

u0 = [6.0, 6.0, 6.0, 200.0, 0.0, 0.0, 0.0]
params = [0.3, 0.45, 0.006, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]

# Shi timing parameters
Eshift = 0.0
Eₘᵢₙ = 0.03
τₑₛ = 0.3
τₑₚ = 0.45
Eₘₐₓ = 1.5
Rmv = 0.006
τ = 1.0

NN = Lux.Chain(
    Lux.Dense(7, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 7),
)

function NIK!(du, u, p, t)
    pLV, psa, psv, Vlv, Qav, Qmv, Qs = u
    τₑₛ, τₑₚ, Rmv, Zao, Rs, Csa, Csv, Eₘₐₓ, Eₘᵢₙ = p
    # The differential equations
    du[1] =
        (Qmv - Qav) * ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) +
        pLV / ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) *
        DShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)
    # 1 Left Ventricle
    du[2] = (Qav - Qs) / Csa #Systemic arteries     
    du[3] = (Qs - Qmv) / Csv # Venous
    du[4] = Qmv - Qav # LV volume
    du[5] = Valve(Zao, (du[1] - du[2]), u[1] - u[2])  # AV 
    du[6] = Valve(Rmv, (du[3] - du[1]), u[3] - u[1])  # MV
    du[7] = (du[2] - du[3]) / Rs # Systemic flow
end

ode_problem = ODEProblem(NIK!, u0, tspan, params)

nn_weight = 0.2
# good results 0.2, 100 iters and all data_vars

trained_params, trained_state = LibInfuser.PINN_Infuser(
    ode_problem,
    NN,
    original_data;
    early_stopping = false,
    nn_output_weight = nn_weight,
    physics_weight = 1.0,
    optimizer = ADAM,
    learning_rate = 0.0002,
    reltol = 1e-6,
    abstol = 1e-6,
    iters = 100,
    rng = rng,
    loss_logfile = "training_logs/loss_history.txt",
    data_vars = [1, 2, 3, 4],
    physics_vars = [5, 6, 7],
)

# LibInfuser.PINN_Symbolic_Regressor(
#     NN,
#     (PINN_solu, trained_st)
# )

extrapolation_tspan = (0.0, 12.5)
extrapolation_num_of_samples = Int(num_of_samples * 6.25)
new_tseps = range(
    extrapolation_tspan[1],
    extrapolation_tspan[2],
    length = extrapolation_num_of_samples,
)

csv_file = "extrapolation.csv"
LibInfuser.PINN_Extrapolator(
    ode_problem,
    extrapolation_tspan,
    original_data,
    nn_weight,
    extrapolation_num_of_samples,
    NN,
    (trained_params, trained_state),
    csv_file,
)

pred_mat = readdlm(csv_file, ',', Float64)

# Get standard ODE solution
prob_ODE = ODEProblem((du, u, p, t) -> NIK!(du, u, p, t), u0, extrapolation_tspan, params)
sol_ODE = solve(prob_ODE, Vern7(), saveat = new_tseps)
ode_solution = hcat(sol_ODE.u...)'

# Define base variable names
variables = ["pLV", "psa", "psv", "Vlv", "Qav", "Qmv", "Qs"]

# Create descriptions with pattern: pinn, data, ode for each variable
descriptions = String[]
for var in variables
    push!(descriptions, "$(var) data")
    push!(descriptions, "$(var) PINN")
    push!(descriptions, "$(var) ODE")
end

LibInfuser.PINNPlotter.plot_PINN_results(
    pred_mat[start_idx:end, :],
    original_data,
    ode_solution[start_idx:end, :],
    descriptions,
    tsteps,
    new_tseps[start_idx:end],
    "t",
    "value",
    "Cardio vascular model comparison",
    "cvs.png",
)
