using OrdinaryDiffEq
using Lux, Plots, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers
using ForwardDiff

rng = StableRNG(5958)
include("../src/lib.jl")

α = 1.1
β = 0.4
δ = 0.1
γ = 0.4
u0 = [10.0, 5.0]
tspan = (0.0, 20.0)
num_of_samples = 300
tsteps = range(tspan[1], tspan[2], length=num_of_samples)

function lv_with_season!(du, u, p, t)
    x, y = u
    α_season = α * (1.0 + 0.3*sin(2π/10))
    du[1] = α_season*x - β*x*y
    du[2] = δ*x*y - γ*y
end

prob_true = ODEProblem(lv_with_season!, u0, tspan)
sol_true = solve(prob_true, Tsit5(), saveat=tsteps)
σ_noise = 0.2
data_noisy = sol_true.u .+ σ_noise .* randn.(size.(sol_true.u))
data_noisy_mat = hcat(data_noisy...)'

NN = Lux.Chain(
    Lux.Dense(2, 20, elu),
    Lux.Dense(20, 20, elu),
    Lux.Dense(20, 2)
)

function lv_to_infuse!(du, u, p, t)
    x, y = u
    du[1] = (α*x - β*x*y)
    du[2] = (δ*x*y - γ*y)
end

infusing_problem = ODEProblem(lv_to_infuse!, u0, tspan)

(PINN_solu, trained_st) = LibInfuser.PINN_Infuser(
    infusing_problem,
    NN,
    data_noisy_mat,
    iters=1
)

LibInfuser.PINN_Symbolic_Regressor(
    NN,
    (PINN_solu, trained_st)
)

# end of training, rest is plotting

function lv_to_infuse2!(du, u, p, t)
    x, y = u
    nn_output = NN(u, PINN_solu, trained_st)[1]
    du[1] = (α*x - β*x*y) * (1 + sin(3.14 * nn_output[1]))
    du[2] = (δ*x*y - γ*y) * (1 + sin(3.14 * nn_output[2]))
end

infusing_problem2 = ODEProblem(lv_to_infuse2!, u0, tspan)

extrapolation_tspan = (0.0, 60.0)
new_tseps = range(extrapolation_tspan[1], extrapolation_tspan[2], length=num_of_samples * 3)

# Get true solution
prob_true_extrapolation = ODEProblem(lv_with_season!, u0, extrapolation_tspan)
sol_true_extrapolation = solve(prob_true_extrapolation, Tsit5(), saveat=new_tseps)
u_true_mat = hcat(sol_true_extrapolation.u...)'

# Get PINN prediction
trained_extrapolation = ODEProblem(lv_to_infuse2!, u0, extrapolation_tspan)
solved_trained_extrapolation = solve(trained_extrapolation, Tsit5(), saveat=new_tseps)
pred_mat = hcat(solved_trained_extrapolation.u...)'

# Get standard ODE solution
prob_ODE = ODEProblem((du,u,p,t)->(du[1]=α*u[1]-β*u[1]*u[2]; du[2]=δ*u[1]*u[2]-γ*u[2]), u0, extrapolation_tspan)
sol_ODE = solve(prob_ODE, Tsit5(), saveat=new_tseps)

# Plotting
plot(new_tseps, u_true_mat[:,1], label="Prey ground truth", lw=2, ls=:dot, color=:blue)
plot!(new_tseps, pred_mat[:,1], label="PINN Prey", lw=3, color=:red)
plot!(new_tseps, sol_ODE[1,:], label="ODE Prey (no NN)", lw=2, ls=:dash, color=:green)

plot!(new_tseps, u_true_mat[:,2], label="Predator ground truth", lw=2, ls=:dot, color=:cyan)
plot!(new_tseps, pred_mat[:,2], label="PINN Predator", lw=3, color=:orange)
plot!(new_tseps, sol_ODE[2,:], label="ODE Predator (no NN)", lw=2, ls=:dash, color=:purple)

xlabel!("t")
ylabel!("Population")
title!("Lotka-Volterra: PINN vs Noisy Data vs Ideal ODE extrapolation")
name = "lotka_normal.png"
savefig("$name")
println("Plot saved as $name")