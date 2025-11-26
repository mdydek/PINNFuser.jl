using OrdinaryDiffEq
using Lux, Plots, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers
using ForwardDiff

rng = StableRNG(5958)

# ------------------------------
# Lotka-Volterra parameters (idealny model)
# ------------------------------
α = 1.1   # growth rate of prey
β = 0.4   # predation rate
δ = 0.1   # reproduction rate of predator per prey eaten
γ = 0.4   # predator death rate

u0 = [10.0, 5.0]  # initial populations: [prey, predator]

# Training range
tspan = (0.0, 20.0)
num_of_samples = 300
tsteps = range(tspan[1], tspan[2], length = num_of_samples)

# ------------------------------
# Generate synthetic "experimental" data
# Introduce a hidden seasonal effect on prey growth
# ------------------------------
function lv_with_season!(du, u, p, t)
    x, y = u
    α_season = α * (1.0 + 0.3 * sin(2pi / 3)) # hidden seasonality
    du[1] = α_season * x - β * x * y
    du[2] = δ * x * y - γ * y
end

prob_true = ODEProblem(lv_with_season!, u0, tspan)
sol_true = solve(prob_true, Tsit5(), saveat = tsteps)

# Add small measurement noise
σ_noise = 0.2
data_noisy = sol_true.u .+ σ_noise .* randn.(size.(sol_true.u))
data_noisy_mat = hcat(data_noisy...)'

# ------------------------------
# Neural network (PINN)
# ------------------------------
NN = Lux.Chain(Lux.Dense(2, 20, elu), Lux.Dense(20, 20, elu), Lux.Dense(20, 2))

p, st = Lux.setup(rng, NN)
p = 0 * ComponentVector{Float64}(p)

# ------------------------------
# PINN ODE (ideal Lotka-Volterra + NN correction)
# ------------------------------
function lv_PINN!(du, u, p, t)
    x, y = u
    NN_output = NN(u, p, st)[1]
    du[1] = (α * x - β * x * y) * (1 + sin(pi * NN_output[1]))
    du[2] = (δ * x * y - γ * y) * (1 + sin(pi * NN_output[2]))
end

prob_PINN = ODEProblem(lv_PINN!, u0, tspan)

# ------------------------------
# Prediction function
# ------------------------------
function predict(p, tspan = tspan, tsteps = tsteps)
    temp_prob = remake(prob_PINN, p = p, tspan = tspan)
    temp_sol = solve(temp_prob, Tsit5(), saveat = tsteps, reltol = 1e-6, abstol = 1e-6)
    return temp_sol
end

# ------------------------------
# Data loss
# ------------------------------
function data_loss(pred, data)
    pred_mat = hcat(pred.u...)'
    return mean(abs2, pred_mat .- data)
end

# ------------------------------
# Physics loss
# ------------------------------
function physics_loss(pred, p)
    pred_mat = hcat(pred.u...)'
    loss = 0.0
    for (i, t) in enumerate(tsteps)
        u = pred_mat[i, :]
        du = similar(u)
        lv_PINN!(du, u, p, t)
        f_standard = [α * u[1] - β * u[1] * u[2], δ * u[1] * u[2] - γ * u[2]]
        loss += sum(abs2.(du .- f_standard))
    end
    return loss / length(tsteps)
end

# ------------------------------
# Total loss
# ------------------------------
function total_loss(p, λ = 1.0)
    pred = predict(p)
    L_data = data_loss(pred, data_noisy_mat)
    L_phys = physics_loss(pred, p)
    return L_data + λ * L_phys
end

# ------------------------------
# Training
# ------------------------------
losses = Float64[]
callback = function (p, l)
    push!(losses, l)
    println("Iteration $(length(losses)): Loss = $(losses[end])")
    return false
end

adtype = Optimization.AutoForwardDiff()
optf = Optimization.OptimizationFunction((x, p) -> total_loss(x, 1.0), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

# --- Training ---
w1 = Optimization.solve(optprob, ADAM(0.01), callback = callback, maxiters = 50)
optprob1 = Optimization.OptimizationProblem(optf, w1.u)
PINN_sol = Optimization.solve(optprob1, ADAM(0.001), callback = callback, maxiters = 100)

# ------------------------------
# Predictions: PINN vs ideal solver vs noisy data
# ------------------------------
extrapolation_tspan = (0.0, 60.0)
new_tseps =
    range(extrapolation_tspan[1], extrapolation_tspan[2], length = num_of_samples * 3)

prob_true_extrapolation = ODEProblem(lv_with_season!, u0, extrapolation_tspan)
sol_true_extrapolation = solve(prob_true_extrapolation, Tsit5(), saveat = new_tseps)
u_true_mat = hcat(sol_true_extrapolation.u...)'

prediction = predict(PINN_sol.u, extrapolation_tspan, new_tseps)
pred_mat = hcat(prediction.u...)'

prob_ODE = ODEProblem(
    (du, u, p, t) ->
        (du[1] = α * u[1] - β * u[1] * u[2]; du[2] = δ * u[1] * u[2] - γ * u[2]),
    u0,
    extrapolation_tspan,
)
sol_ODE = solve(prob_ODE, Tsit5(), saveat = new_tseps)

plot(new_tseps, u_true_mat[:, 1], label = "Prey ground truth", lw = 2, ls = :dot)
plot!(new_tseps, pred_mat[:, 1], label = "PINN Prey", lw = 3)
plot!(new_tseps, sol_ODE[1, :], label = "ODE Prey (no NN)", lw = 2, ls = :dash)
plot(new_tseps, u_true_mat[:, 2], label = "Predator ground truth", lw = 2, ls = :dot)
plot!(new_tseps, pred_mat[:, 2], label = "PINN Predator", lw = 3)
plot!(new_tseps, sol_ODE[2, :], label = "ODE Predator (no NN)", lw = 2, ls = :dash)
xlabel!("t")
ylabel!("Population")
title!("Lotka-Volterra: PINN vs Noisy Data vs Ideal ODE extrapolation")
savefig("experiments/lotka_volterra_plots/pinn_lv_seasonal_plot_extrapolation.png")
println("Plot saved as pinn_lv_seasonal_plot_extrapolation.png")
