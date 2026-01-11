using OrdinaryDiffEq
using Lux, Plots, Zygote, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Optim, Measures, BenchmarkTools
using DelimitedFiles, ForwardDiff

include("./simple_CVS_model.jl")
using .simpleModel

include("../../src/lib.jl")
using .LibInfuser

# Training range
tspan = (0.0, 7.0)
num_of_samples = 300
tsteps = range(5.0, 7.0, length = num_of_samples)

loaded_data = readdlm("examples/OneChamberModelCVS/original_data.txt")
extrap_original_data = Array{Float64}(loaded_data)[1:3000, :]
original_data = extrap_original_data[751:1050, :]

u0 = [6.0, 6.0, 6.0, 200.0, 0.0, 0.0, 0.0]
params = [0.3, 0.45, 0.006, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]
τₑₛ, τₑₚ, Rmv, Zao, Rs, Csa, Csv, Eₘₐₓ, Eₘᵢₙ = params
Eshift = 0.0
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

    du[1] =
        (Qmv - Qav) * ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) +
        pLV / ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) *
        DShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)
    # Left Ventricle
    du[2] = (Qav - Qs) / Csa #Systemic arteries     
    du[3] = (Qs - Qmv) / Csv # Venous
    du[4] = Qmv - Qav # LV volume
    du[5] = Valve(Zao, (du[1] - du[2]), u[1] - u[2])  # AV 
    du[6] = Valve(Rmv, (du[3] - du[1]), u[3] - u[1])  # MV
    du[7] = (du[2] - du[3]) / Rs # Systemic flow
end

ode_problem = ODEProblem(NIK!, u0, tspan)

trained_p, trained_st = LibInfuser.PINN_Infuser(
    ode_problem,
    NN,
    tsteps,
    original_data;
    early_stopping = true,
    nn_output_weight = 0.1,
    physics_weight = 1.0,
    learning_rate = 0.005,
    reltol = 1e-6,
    abstol = 1e-6,
    dtmax = 1e-2,
    iters = 200,
    loss_logfile = "training_logs/loss_history.txt",
    data_vars = [1, 2, 3, 4],
    physics_vars = [5, 6, 7],
)

# Save extrapolation data
extrapolation_tspan = (0.0, 20.0)
new_tseps = range(extrapolation_tspan[1], extrapolation_tspan[2], length = 3000)

LibInfuser.PINN_Extrapolator(
    ode_problem,
    NN,
    (trained_p, trained_st),
    extrapolation_tspan,
    3000,
    "cvs_lib_extrapolation.txt";
    nn_output_weight = 0.05,
    reltol = 1e-4,
    abstol = 1e-7,
    dtmax = 1e-2,
)

mkpath("plots")

pinn_pred = readdlm("cvs_lib_extrapolation.txt", ',', Float64)
ode_problem_extrap = ODEProblem(NIK!, u0, extrapolation_tspan)

ode_sol = solve(
    ode_problem_extrap,
    Vern7();
    saveat = new_tseps,
    reltol = 1e-6,
    abstol = 1e-6,
)
one_chamber_sol = Matrix(Array(ode_sol)')

labels = [
    "Four chamber pLV", "PINN pLV", "One chamber pLV",
    "Four chamber psa", "PINN psa", "One chamber psa",
    "Four chamber psv", "PINN psv", "One chamber psv",
    "Four chamber Vlv", "PINN Vlv", "One chamber Vlv",
    "Four chamber Qav", "PINN Qav", "One chamber Qav",
    "Four chamber Qmv", "PINN Qmv", "One chamber Qmv",
    "Four chamber Qs", "PINN Qs", "One chamber Qs",
]

mask = new_tseps .>= 5.0

time_to_plot = new_tseps[mask]
pinn_to_plot = pinn_pred[mask, :]
data_to_plot = extrap_original_data[mask, :]
ode_to_plot = one_chamber_sol[mask, :]

LibInfuser.PINNPlotter.plot_PINN_results(
    pinn_to_plot,   # PINN
    data_to_plot,   # Data
    ode_to_plot,    # ODE
    labels,
    time_to_plot,
    time_to_plot,
    "Time [s]",
    "State value",
    "CVS: PINN vs One chamber vs Four_chamber",
    "plots/cvs_comparison.png",
)

LibInfuser.PINNPlotter.plot_loss(
    "training_logs/loss_history.txt";
    plotfile = "plots/loss.png",
)